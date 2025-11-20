from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="Narendran_Predictive/deployment",     # the local folder containing your files
    # replace with your repoid
    repo_id="Narendranh/narendran_predictive",          # the target repo

    repo_type="space",
    path_in_repo="",
)
