# Set up!

## Activating your environment
module load miniconda3/25.9.1
conda activate eureka
cd ~/eureka_project
pip install -e .
pip install openai==0.28 "Pillow<10.0.0" gpustat


<p>Installing Isaac </p>
1. Go to [developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) in your Windows browser
2. Download `IsaacGym_Preview_4_Package.tar.gz` (free NVIDIA account required)
3. Run this on your local Windows machine (PowerShell or Command Prompt)
scp C:\Users\<your-username>\Downloads\IsaacGym_Preview_4_Package.tar.gz <your-uid>@notchpeak.chpc.utah.edu:~/
4. Then on CHPC
cd ~
tar -xvf IsaacGym_Preview_4_Package.tar.gz
module load miniconda3/25.9.1
conda activate eureka
cd isaacgym/python
pip install -e .

<p> More dependenciies </p>
cd ~eureka projec dir
cd isaacgymenvs && pip install -e .
cd ../rl_games && pip install -e .

<p>fix dependency issues </p>

pip install openai==0.28
pip install "Pillow<10.0.0"
pip install gpustat

Save your open ai key:
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc


Update "recipient" to your email in /eureka/cfg/chpc_alerts.cfg

# To Run
sbatch job_scipts/run.sh

#While Running
To give feedback:
echo "the reward function is too sparse, penalize early termination" > ~/eureka_project/eureka/human_feedback.txt