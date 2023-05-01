import argparse
import os
import subprocess
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


if __name__ == "__main__":

    import evalresults

    print("Testing MOOD docker image...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--docker_name", required=True, type=str, help="Name of the docker image you want to test"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=str,
        help=(
            "Data dir, it will require to contain a folder 'brain' and 'abdom' which will both "
            "each require a subfolder 'toy' and 'toy_label' i.e. data_dir/brain/toy,"
            " data_dir/brain/toy_label, data_dir/abdom/toy, data_dir/abdom/toy_label"
        ),
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, type=str, help="Folder where the output/ predictions will be written too"
    )
    parser.add_argument(
        "-t", "--task", required=True, choices=["sample", "pixel"], type=str, help="Task, either 'pixel' or 'sample' "
    )
    parser.add_argument(
        "--no_gpu",
        required=False,
        default=False,
        type=bool,
        help="If you have not installed the nvidia docker toolkit, set this arg to False",
    )

    args = parser.parse_args()

    docker_name = args.docker_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    task = args.task
    no_gpu = args.no_gpu

    tmp_dir = None
    if output_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()
        output_dir = tmp_dir.name

    brain_data_dir = os.path.join(input_dir, 'imgs')
    brain_data_label_dir = os.path.join(input_dir, 'labels')

    output_brain_dir = os.path.join(output_dir, "brain")
    os.makedirs(output_brain_dir, exist_ok=True)

    gpu_str = ""
    if no_gpu:
        gpu_str = "--gpus device=1 "

    print("\nPredicting SISS2015 data...")

    ret = ""
    try:
        docker_str = (
            f"docker run {gpu_str}-v {brain_data_dir}:/mnt/data "
            f"-v {output_brain_dir}:/mnt/pred --read-only {docker_name} sh /workspace/run_{task}_brain.sh /mnt/data /mnt/pred"
        )
        ret = subprocess.run(docker_str.split(" "), check=True,)
    except Exception:
        print(f"Running Docker brain-{task}-script failed:")
        print(ret)
        exit(1)

    print("\nEvaluating predictions...")

    brain_score = evalresults.eval_dir(
        output_brain_dir, brain_data_label_dir, mode=task, save_file=os.path.join(output_dir, "brain_score.txt")
    )
    print("Brain-dataset score:", brain_score)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    print("Done.")
