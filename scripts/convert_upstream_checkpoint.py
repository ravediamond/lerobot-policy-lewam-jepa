import argparse

from lerobot_policy_lewam_jepa.convert import convert


def main():
    p = argparse.ArgumentParser(
        description="Convert a LeWAM checkpoint (lewam_best.pt + lewam_config.json) to lewam_jepa"
    )
    p.add_argument("--checkpoint", help="path to lewam_best.pt")
    p.add_argument("--config", help="path to lewam_config.json")
    p.add_argument("--repo-id", help="Hub repo holding both files, e.g. LeWAM/lewam-pusht")
    p.add_argument("--revision", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--image-key", default="observation.image")
    p.add_argument("--goal-image-key", default="observation.images.goal")
    a = p.parse_args()
    provenance = {}
    if a.repo_id:
        from huggingface_hub import HfApi, hf_hub_download

        revision = a.revision or HfApi().model_info(a.repo_id).sha
        a.checkpoint = hf_hub_download(a.repo_id, "lewam_best.pt", revision=revision)
        a.config = hf_hub_download(a.repo_id, "lewam_config.json", revision=revision)
        provenance = {"repo_id": a.repo_id, "revision": revision}
    if not (a.checkpoint and a.config):
        p.error("give --repo-id, or --checkpoint and --config")
    convert(a.checkpoint, a.config, a.out, a.image_key, a.goal_image_key, provenance=provenance)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
