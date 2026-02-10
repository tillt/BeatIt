import argparse
import inspect
from pathlib import Path

import torch
import coremltools as ct

from beat_this.model.beat_tracker import BeatThis
from beat_this.utils import replace_state_dict_key

CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"


class BeatThisCoreMLWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, x: torch.Tensor):
        # x shape: (B, T, F)
        output_dict = self.model(x)
        beat_logits = output_dict["beat"].squeeze(0)
        downbeat_logits = output_dict["downbeat"].squeeze(0)
        return beat_logits, downbeat_logits


def resolve_checkpoint(model_name: str, ckpt_dir: Path) -> Path | str:
    if Path(model_name).exists():
        return Path(model_name)
    candidate = ckpt_dir / f"beat_this-{model_name}.ckpt"
    if candidate.exists():
        return candidate
    candidate = ckpt_dir / f"{model_name}.ckpt"
    if candidate.exists():
        return candidate
    return model_name


def load_model_local(checkpoint_path: Path | str, device: str | torch.device = "cpu") -> BeatThis:
    weights_only = {"weights_only": True} if torch.__version__ >= "2" else {}
    if isinstance(checkpoint_path, Path) and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, **weights_only)
    else:
        if not (
            str(checkpoint_path).startswith("https://")
            or str(checkpoint_path).startswith("http://")
        ):
            checkpoint_url = f"{CHECKPOINT_URL}/{checkpoint_path}.ckpt"
            file_name = f"beat_this-{checkpoint_path}.ckpt"
        else:
            checkpoint_url = checkpoint_path
            file_name = None
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_url, file_name=file_name, map_location=device
        )

    hparams = checkpoint["hyper_parameters"]
    hparams = {k: v for k, v in hparams.items() if k in set(inspect.signature(BeatThis).parameters)}
    model = BeatThis(**hparams)
    state_dict = replace_state_dict_key(checkpoint["state_dict"], "model.", "")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def convert_one(model_name: str,
                out_dir: Path,
                time_steps: int,
                n_mels: int,
                deployment: str,
                precision: str,
                convert_to: str,
                ckpt_dir: Path):
    print(f"\n=== {model_name} ===")
    print(">>> Loading model...")
    ckpt = resolve_checkpoint(model_name, ckpt_dir)
    pt_model = load_model_local(ckpt, device="cpu")
    wrapper = BeatThisCoreMLWrapper(pt_model)

    dummy_input = torch.randn(1, time_steps, n_mels)

    print(">>> Warming up model cache...")
    with torch.no_grad():
        _ = wrapper(dummy_input)

    print(">>> Tracing model...")
    traced_model = torch.jit.trace(wrapper, dummy_input, check_trace=False)

    print(">>> Converting to Core ML...")
    compute_precision = None
    if convert_to == "mlprogram":
        compute_precision = ct.precision.FLOAT32 if precision == "float32" else ct.precision.FLOAT16
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="mel_spectrogram", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="beat_logits"), ct.TensorType(name="downbeat_logits")],
        minimum_deployment_target=getattr(ct.target, deployment),
        compute_precision=compute_precision,
        convert_to=convert_to,
    )

    mlmodel.author = "CPJKU (Converted for iOS)"
    mlmodel.license = "MIT"
    mlmodel.short_description = f"Beat This! ({model_name}) - Beat Tracking"
    mlmodel.input_description["mel_spectrogram"] = "Log Mel Spectrogram (Time x 128)."

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"BeatThis_{model_name}.mlpackage"
    mlmodel.save(save_path)
    print(f">>> Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert BeatThis checkpoints to Core ML.")
    parser.add_argument("--models", nargs="+", default=["final0", "final1", "final2", "small0", "small1", "small2"])
    parser.add_argument("--out", default="coreml_out")
    parser.add_argument("--time-steps", type=int, default=1500)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--ckpt-dir", default="models")
    parser.add_argument("--deployment", default="macOS13",
                        choices=["iOS16", "iOS17", "iOS18", "macOS11", "macOS13", "macOS14"])
    parser.add_argument("--precision", default="float32", choices=["float32", "float16"])
    parser.add_argument("--convert-to", default="mlprogram", choices=["mlprogram", "neuralnetwork"])
    args = parser.parse_args()

    out_dir = Path(args.out)
    ckpt_dir = Path(args.ckpt_dir)
    print(f"Output: {out_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Input:  (1, {args.time_steps}, {args.n_mels})")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Target: {args.deployment}, precision={args.precision}, convert_to={args.convert_to}")

    for model_name in args.models:
        convert_one(model_name,
                    out_dir,
                    args.time_steps,
                    args.n_mels,
                    args.deployment,
                    args.precision,
                    args.convert_to,
                    ckpt_dir)


if __name__ == "__main__":
    main()
