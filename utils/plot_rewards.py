import atexit
import multiprocessing as mp
from collections import deque


def _plot_worker(recv_conn, max_points=1000, update_interval_ms=250):
    """Run matplotlib GUI in a dedicated process so training never blocks the UI."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    epochs = deque(maxlen=max_points)
    rewards = deque(maxlen=max_points)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Training Reward Over Time", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Reward")
    ax.grid(True, alpha=0.3)
    (line,) = ax.plot([], [], "b-o", linewidth=0.5, markersize=1)

    def on_close(_event):
        try:
            recv_conn.close()
        except OSError:
            pass

    fig.canvas.mpl_connect("close_event", on_close)

    def update(_frame):
        # Drain all pending updates quickly.
        while recv_conn.poll():
            msg = recv_conn.recv()
            if msg == "__close__":
                plt.close(fig)
                return (line,)
            # Support a save command sent from the parent process: ("__save__", path)
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__save__":
                try:
                    fig.savefig(msg[1])
                except Exception:
                    pass
                continue
            # Message formats supported:
            # (epoch, reward) or (epoch, raw_reward, norm_reward)
            if isinstance(msg, tuple) and len(msg) == 2:
                epoch, reward = msg
                norm_reward = None
            elif isinstance(msg, tuple) and len(msg) >= 3:
                epoch, raw_reward, norm_reward = msg[0], msg[1], msg[2]
                reward = norm_reward
            else:
                # Fallback: unexpected format
                try:
                    epoch, reward = msg
                    norm_reward = None
                except Exception:
                    continue
            epochs.append(epoch)
            rewards.append(reward)

        if rewards:
            x = list(epochs)
            y = list(rewards)
            line.set_data(x, y)

            min_reward = min(y)
            max_reward = max(y)
            reward_range = max_reward - min_reward
            padding = reward_range * 0.1 if reward_range > 0 else max(abs(max_reward) * 0.1, 1.0)

            ax.set_ylim(min_reward - padding, max_reward + padding)
            ax.set_xlim(-1, max(x) + 1)

        return (line,)

    try:
        ani = FuncAnimation(
            fig,
            update,
            interval=update_interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        # On some macOS/TkAgg setups a FuncAnimation can be created but have
        # `event_source` set to None which later causes `add_callback` to fail
        # during the draw step. Detect that case and treat it as a setup
        # failure so we use the manual update fallback below.
        if getattr(ani, "event_source", None) is None:
            raise RuntimeError("FuncAnimation.event_source is None")
        # Keep a strong reference so animation callbacks continue to run.
        fig._reward_animation = ani
    except (AttributeError, RuntimeError) as e:
        # FuncAnimation failed (common on macOS TkAgg). Fall back to periodic manual updates.
        import sys
        print(f"Warning: FuncAnimation setup failed ({e}). Using manual updates instead.", file=sys.stderr)
        def manual_update():
            while recv_conn.poll():
                msg = recv_conn.recv()
                if msg == "__close__":
                    plt.close(fig)
                    return
                # Handle save command
                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__save__":
                    try:
                        fig.savefig(msg[1])
                    except Exception:
                        pass
                    continue
                # Message formats supported: (epoch, reward) or (epoch, raw, norm)
                if isinstance(msg, tuple) and len(msg) == 2:
                    epoch, reward = msg
                elif isinstance(msg, tuple) and len(msg) >= 3:
                    epoch, raw_reward, reward = msg[0], msg[1], msg[2]
                else:
                    try:
                        epoch, reward = msg
                    except Exception:
                        continue
                epochs.append(epoch)
                rewards.append(reward)

            if rewards:
                x = list(epochs)
                y = list(rewards)
                line.set_data(x, y)
                min_reward = min(y)
                max_reward = max(y)
                reward_range = max_reward - min_reward
                padding = reward_range * 0.1 if reward_range > 0 else max(abs(max_reward) * 0.1, 1.0)
                ax.set_ylim(min_reward - padding, max_reward + padding)
                ax.set_xlim(-1, max(x) + 1)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("draw_event", lambda e: manual_update())

    plt.show()


class RewardPlotter:
    """Real-time reward plotting in a separate process."""

    def __init__(self, max_points=1000, update_interval_ms=250):
        self._epoch = 0
        self._closed = False

        # Use spawn on macOS to avoid inheriting unstable GUI state.
        self._ctx = mp.get_context("spawn")
        self._recv_conn, self._send_conn = self._ctx.Pipe(duplex=False)
        self._proc = self._ctx.Process(
            target=_plot_worker,
            args=(self._recv_conn, max_points, update_interval_ms),
            daemon=True,
        )
        self._proc.start()

        atexit.register(self.close)

    def add_reward(self, reward):
        if self._closed or not self._proc.is_alive():
            return
        try:
            # Support sending optional normalized reward: caller may pass a
            # tuple (raw, normalized) by calling add_reward(raw, normalized)
            if isinstance(reward, tuple) and len(reward) == 2:
                raw, norm = float(reward[0]), float(reward[1])
                self._send_conn.send((self._epoch, raw, norm))
            else:
                self._send_conn.send((self._epoch, float(reward)))
            self._epoch += 1
        except (BrokenPipeError, EOFError, OSError):
            self._closed = True

    def save(self, path):
        """Request the plot worker save the current figure to `path`.

        The save request is sent to the plotting process which calls
        `fig.savefig(path)`. If the worker is not alive or the pipe is
        closed this becomes a no-op.
        """
        if self._closed:
            return
        try:
            # Send a tuple command that the worker recognizes.
            self._send_conn.send(("__save__", str(path)))
        except (BrokenPipeError, EOFError, OSError):
            self._closed = True

    def save(self, path):
        """Request the plot worker save the current figure to `path`.

        The save request is sent to the plotting process which calls
        `fig.savefig(path)`. If the worker is not alive or the pipe is
        closed this becomes a no-op.
        """
        if self._closed:
            return
        try:
            # Send a tuple command that the worker recognizes.
            self._send_conn.send(("__save__", str(path)))
        except (BrokenPipeError, EOFError, OSError):
            self._closed = True

    def close(self):
        if self._closed:
            return
        self._closed = True

        try:
            if self._proc.is_alive():
                self._send_conn.send("__close__")
        except (BrokenPipeError, EOFError, OSError):
            pass

        try:
            self._send_conn.close()
        except OSError:
            pass

        if self._proc.is_alive():
            self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
