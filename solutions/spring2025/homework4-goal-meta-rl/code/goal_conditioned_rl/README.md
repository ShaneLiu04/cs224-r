
# Setup
Make sure that you have used the given AMI to create a c4.4xlarge instance with 60gb of storage. Follow the CS 224R AWS Guide which has instructions for setting up and accessing an AWS spot instance.

SSH into the virtual machine. There should already be a hw4 folder in the default directory. Conda should already be installed and you should see the "(base)" text on the left of the terminal.

To get working on the goal_conditioned portion you just need to activate the respective gcrl conda environment
```bash
conda deactivate
conda activate gcrl
```