
ssh ps2622@login.cx3.hpc.imperial.ac.uk
When logged in from main BoCOli folder
# To download from HPC to local machine
scp -r ps2622@login.cx3.hpc.imperial.ac.uk:/rds/general/user/ps2622/home/projects/imperial/data/bayes_sim ./data/cd


# To load everything to ssh
From parent directory of bocoli
scp -r imperial ps2622@login.cx3.hpc.imperial.ac.uk:~/projects/

# to load just the h6.file
From project dir

scp h6_simulations.py ps2622@login.cx3.hpc.imperial.ac.uk:~/projects/imperial/h6_simulations.py

scp -r PBS/* ps2622@login.cx3.hpc.imperial.ac.uk:~/projects/imperial/PBS/



Newest run:
1023253.pbs-7
