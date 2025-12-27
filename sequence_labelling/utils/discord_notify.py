import json
import os
import time
import urllib.request
import statistics
from typing import Any, Dict, Optional

from transformers import TrainerCallback


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int = 10) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "MailEX-DiscordNotify"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        # Discord returns 204 No Content on success
        _ = resp.read()


def send_discord_message(webhook_url: str, content: str) -> None:
    if not webhook_url:
        return
    # Discord content limit is 2000 chars; truncate defensively.
    content = content[:1990]
    _post_json(webhook_url, {"content": content})


class DiscordNotifyEveryNEpochsCallback(TrainerCallback):
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        *,
        every_n_epochs: int = 10,
        preferred_prf_prefixes: Optional[tuple[str, ...]] = None,
        preferred_score_keys: Optional[tuple[str, ...]] = None,
        run_name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
        self.every_n_epochs = every_n_epochs
        self.preferred_prf_prefixes = preferred_prf_prefixes or (
            "eval_EM_trigger_class_scores",
            "eval_EM_arg_class_scores",
            "eval_f_score_class",
            "eval_f_score_id",
        )
        self.preferred_score_keys = preferred_score_keys or (
            "eval_EM_trigger_class_scores_F1",
            "eval_EM_arg_class_scores_F1",
            "eval_f_score_class",
            "eval_f_score_id",
        )
        self.run_name = run_name
        self.enabled = enabled and bool(self.webhook_url)
        self._sent_epochs = set()
        self._last_logs: Dict[str, Any] = {}
        self._train_start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._epoch_durations_s: list[float] = []

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f"eta={seconds:.0f}s"
        minutes = seconds / 60.0
        if minutes < 60:
            return f"eta={minutes:.1f}m"
        hours = int(minutes // 60)
        rem_min = int(round(minutes - hours * 60))
        return f"eta={hours}h{rem_min:02d}m"

    def _format_header(self, args, epoch_int: Optional[int], prefix: str) -> str:
        total_epochs = getattr(args, "num_train_epochs", None)
        total_epochs_int = None
        if total_epochs is not None:
            try:
                total_epochs_int = int(round(float(total_epochs)))
            except Exception:
                total_epochs_int = None

        if epoch_int is not None and total_epochs_int and total_epochs_int > 0:
            progress_pct = min(100.0, (epoch_int / total_epochs_int) * 100.0)
            header = f"{prefix}: epoch {epoch_int}/{total_epochs_int} ({progress_pct:.1f}%)"
        elif epoch_int is not None:
            header = f"{prefix}: epoch {epoch_int}"
        else:
            header = f"{prefix}"

        if self.run_name:
            header = f"{self.run_name}: {header}"
        return header

    def _extract_score_str(self) -> Optional[str]:
        score_key = next((k for k in self.preferred_score_keys if k in self._last_logs), None)
        if score_key is None:
            return None
        try:
            score_val = float(self._last_logs[score_key])
            return f"{score_key}={score_val:.6f}"
        except Exception:
            return f"{score_key}={self._last_logs[score_key]}"

    def _extract_prf_str(self) -> Optional[str]:
        # Prefer full Precision/Recall/F1 triples if available.

        def _display_key(k: str) -> str:
            # Keep official metric keys, but avoid abbreviated label in the message.
            # (e.g., "..._F1" -> "..._F1Score")
            if k.endswith("_F1"):
                return k[:-3] + "_F1Score"
            return k

        for prefix in self.preferred_prf_prefixes:
            if prefix in ("eval_f_score_class", "eval_f_score_id"):
                # These metrics historically exist as a single F-score only.
                if prefix in self._last_logs:
                    try:
                        f1 = float(self._last_logs[prefix])
                        return f"{_display_key(prefix)}={f1:.6f}"
                    except Exception:
                        return f"{_display_key(prefix)}={self._last_logs[prefix]}"
                continue

            k_p = f"{prefix}_Precision"
            k_r = f"{prefix}_Recall"
            k_f1 = f"{prefix}_F1"
            if k_p not in self._last_logs and k_r not in self._last_logs and k_f1 not in self._last_logs:
                continue

            parts: list[str] = []
            if k_p in self._last_logs:
                try:
                    parts.append(f"{_display_key(k_p)}={float(self._last_logs[k_p]):.6f}")
                except Exception:
                    parts.append(f"{_display_key(k_p)}={self._last_logs[k_p]}")
            if k_r in self._last_logs:
                try:
                    parts.append(f"{_display_key(k_r)}={float(self._last_logs[k_r]):.6f}")
                except Exception:
                    parts.append(f"{_display_key(k_r)}={self._last_logs[k_r]}")
            if k_f1 in self._last_logs:
                try:
                    parts.append(f"{_display_key(k_f1)}={float(self._last_logs[k_f1]):.6f}")
                except Exception:
                    parts.append(f"{_display_key(k_f1)}={self._last_logs[k_f1]}"
                    )

            if parts:
                return " ".join(parts)

        # Fallback to single score string.
        return self._extract_score_str()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self._last_logs.update(logs)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start_time = time.time()
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        self._train_start_time = time.time()
        msg = self._format_header(args, 0, "train start")
        try:
            send_discord_message(self.webhook_url, msg)
        except Exception:
            pass
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return control

        epoch_int = None
        if state.epoch is not None:
            try:
                epoch_int = int(round(state.epoch))
            except Exception:
                epoch_int = None

        parts = [self._format_header(args, epoch_int, "train end")]
        prf_str = self._extract_prf_str()
        if prf_str:
            parts.append(prf_str)
        if self._train_start_time is not None:
            dur_s = max(0.0, time.time() - self._train_start_time)
            parts.append(f"duration={dur_s/60.0:.1f}m")
        msg = " | ".join(parts)

        try:
            send_discord_message(self.webhook_url, msg)
        except Exception:
            pass
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # Always record epoch duration if possible (even when notifications disabled).
        if self._epoch_start_time is not None:
            self._epoch_durations_s.append(max(0.0, time.time() - self._epoch_start_time))
            self._epoch_start_time = None

        if not self.enabled:
            return control
        if state.epoch is None:
            return control

        # HF Trainer uses float epochs; at epoch end it's typically an integer-ish value.
        epoch_int = int(round(state.epoch))
        if epoch_int <= 0:
            return control
        if epoch_int % self.every_n_epochs != 0:
            return control
        if epoch_int in self._sent_epochs:
            return control

        parts = [self._format_header(args, epoch_int, "progress")]

        prf_str = self._extract_prf_str()
        if prf_str:
            parts.append(prf_str)

        # ETA based on average of recent epoch durations.
        total_epochs = getattr(args, "num_train_epochs", None)
        total_epochs_int: Optional[int] = None
        if total_epochs is not None:
            try:
                total_epochs_int = int(round(float(total_epochs)))
            except Exception:
                total_epochs_int = None

        if total_epochs_int and total_epochs_int > 0 and self._epoch_durations_s:
            recent = self._epoch_durations_s[-5:]
            avg_epoch_s = statistics.mean(recent)
            remaining_epochs = max(0, total_epochs_int - epoch_int)
            eta_s = remaining_epochs * avg_epoch_s
            parts.append(self._format_eta(eta_s))

        msg = " | ".join(parts)
        try:
            send_discord_message(self.webhook_url, msg)
            self._sent_epochs.add(epoch_int)
        except Exception:
            self._sent_epochs.add(epoch_int)
        return control
