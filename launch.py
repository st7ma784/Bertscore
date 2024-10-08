
import pytorch_lightning
from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping
import os,sys
import wandb
from HFSummarisationDataModuleExample import MyDataModule
from models.train import myLightningModule
from transformers import AutoModel, AutoTokenizer

#### This is our launch function, which builds the dataset, and then runs the model on it. 
def train(config={
        "batch_size":16, # ADD MODEL ARGS HERE
         "codeversion":"-1",        
    },dir=None,devices=None,accelerator=None,Dataset=None,logtool=None):

    #get model name from the config 
    #get the tokenizer from config too
    #initialize the hf model and autotokenizer from the names provided. 

    #pass the model to the lightning module, and the tokenizer to the datamodule
    model=None
    tokenizer=None
    model_name_or_path=config.get("modelname",None)
    if model_name_or_path is not None:
        model = AutoModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)




    if dir is None:
        dir=config.get("dir",".")
    if Dataset is None:
        Dataset=MyDataModule(Cache_dir=dir,tokenizer=tokenizer,**config)
        # Dataset.prepare_data()
        #idf_dict=Dataset.idf_dict
        idf_dict={}
    model=myLightningModule(model=model,idf_dict=idf_dict,tokenizer=tokenizer, **config)

    if devices is None:
        devices=config.get("devices","auto")
    if accelerator is None:
        accelerator=config.get("accelerator","auto")
    # print("Training with config: {}".format(config))
    Dataset.batch_size=config["batch_size"]
    callbacks=[
        TQDMProgressBar(),
    ]

    #workaround for NCCL issues on windows 
    if sys.platform == "win32":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]='gloo'
    trainer=pytorch_lightning.Trainer(
            devices=1,
            accelerator=accelerator,
            max_epochs=1,
            # max_test_steps=50,
            #profiler="advanced",
            logger=logtool,
            strategy="ddp",
            num_nodes=int(os.getenv("SLURM_NNODES",1)),
            callbacks=callbacks,
            fast_dev_run=False,
            precision=32
    )
        
    trainer.test(model,Dataset)
    wandb.finish() # Finish any old runs
    return
#### This is a wrapper to make sure we log with Weights and Biases, You'll need your own user for this. 
def wandbtrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None):
    if config is not None:
        import wandb
        config=config.__dict__
        dir=config.get("dir",dir)
        wandb.login(key='9cf7e97e2460c18a89429deed624ec1cbfb537bc')
        wandb.finish() # Finish any old runs
        run=wandb.init(project="BertCLIPFP8SUMScore",entity="st7ma784",name="CBertScore",config=config)

        logtool= pytorch_lightning.loggers.WandbLogger( project="BertCLIPFP8SUMScore",entity="st7ma784",experiment=run, save_dir=dir)
        print(config)

    else: 
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        import wandb
        print("here")
        wandb.login(key='9cf7e97e2460c18a89429deed624ec1cbfb537bc')
        run=wandb.init(project="BertCLIPFP8SUMScore",entity="st7ma784",name="CBertScore",config=config)
        logtool= pytorch_lightning.loggers.WandbLogger( project="BertCLIPFP8SUMScore",entity="st7ma784",experiment=run, save_dir=dir)
        config=run.config.as_dict()
    
    train(config,dir,devices,accelerator,Dataset,logtool)
def SlurmRun(trialconfig):

    job_with_version = '{}v{}'.format("bertscore", 0)

    sub_commands =['#!/bin/bash',
        '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',   
        '#SBATCH --time={}'.format( '1:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version), 
        '#SBATCH --nodes=1',  #Nodes per experiment
        '#SBATCH --ntasks-per-node=1',# Set this to GPUs per node. 
        '#SBATCH --gres=gpu:1',  #{}'.format(per_experiment_nb_gpus),
        f'#SBATCH --signal=USR1@{5 * 60}',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('test@gmail.com'),
    ]
    comm="python"
    slurm_commands={}

    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
                '#SBATCH --account bdlan05',
                'export CONDADIR=/nobackup/projects/bdlan05/$USER/miniconda/envs/',
                'export NCCL_SOCKET_IFNAME=ib0',
                'source /nobackup/projects/bdlan05/$USER/miniconda/etc/profile.d/conda.sh',
])
        comm="python3"
    
    elif str(os.getenv("HOSTNAME","localhost")).endswith("hec.lancs.ac.uk"):
        
        sub_commands.extend([
                             '#SBATCH --partition gpu-short',
                            #  '#SBATCH --exclusive',
                            '#SBATCH --mem=64G',
                            '#SBATCH --cpus-per-task=16',
                             'export CONDADIR=$global_storage/conda4',
                             'module add opence',])
        comm="python3"
    else: 
        sub_commands.extend(['export CONDADIR=/home/$USER/miniconda3/envs',
                             'export NCCL_SOCKET_IFNAME=enp0s31f6',])
    sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'conda activate $CONDADIR/open-ce',# ...and activate the conda environment
    ])
    script_name= os.path.realpath(sys.argv[0]) #Find this scripts name...
    trialArgs=__get_hopt_params(trialconfig)
    #If you're deploying prototyping code and often changing your pip env, 
    # consider adding in a 'scopy requirements.txt
    # and then append command 'pip install -r requirements.txt...
    # This should add your pip file from the launch dir to the run location, then install on each node. 
    
    sub_commands.append('srun {} {} {}'.format(comm, script_name,trialArgs))
    #when launched, this script will be called with no trials, and so drop into the wandbtrain section, 
    sub_commands = [x.lstrip() for x in sub_commands]        

    full_command = '\n'.join(sub_commands)
    return full_command

def __get_hopt_params(trial):
    """
    Turns hopt trial into script params
    :param trial:
    :return:
    """
    params = []
    for k in trial.__dict__:
        v = trial.__dict__[k]
        if k == 'num_trials':
            v=0
        # don't add None params
        if v is None or v is False:
            continue

        # put everything in quotes except bools
        if __should_escape(v):
            cmd = '--{} \"{}\"'.format(k, v)
        else:
            cmd = '--{} {}'.format(k, v)
        params.append(cmd)

    # this arg lets the hyperparameter optimizer do its thin
    full_cmd = ' '.join(params)
    return full_cmd

def __should_escape(v):
    v = str(v)
    return '[' in v or ';' in v or ' ' in v
if __name__ == '__main__':
    from demoparse import parser
    from subprocess import call

    myparser=parser()
    hyperparams = myparser.parse_args()

    defaultConfig=hyperparams.__dict__
    
    NumTrials=hyperparams.num_trials
    #BEDE has Env var containing hostname  #HOSTNAME=login2.bede.dur.ac.uk check we arent launching on this node


    if NumTrials ==0 and not str(os.getenv("HOSTNAME","localhost")).startswith("login"): #We'll do a trial run...
        #means we've been launched from a BEDE script, so use config given in args///
        wandbtrain(hyperparams)

    #OR To run with Default Args
    else: 
        trials=myparser.generate_wandb_trials("st7ma784","BertCLIPFP8SUMScore")
        if len(trials)==1:
            
            trial=trials[0]
        #We'll grab a random trial, BUT have to launch it with KWARGS, so that DDP works.       
        #result = call('{} {} --num_trials=0 {}'.format("python",os.path.realpath(sys.argv[0]),__get_hopt_params(trial)), shell=True)

            print("Running trial: {}".format(trial))
            
            wandbtrain(trial)
        else:
            for i,trial in enumerate(trials):             
                command=SlurmRun(trial)
                os.makedirs(os.path.join(".","BERTSCORER"),exist_ok=True)
                slurm_cmd_script_path =  os.path.join(".","BERTSCORER","slurm_cmdtrial{}.sh".format(i))

                with open(slurm_cmd_script_path, "w") as f:
                    f.write(command)
                print('\nlaunching exp...')
                result = call('{} {}'.format("sbatch", slurm_cmd_script_path), shell=True)
                if result == 0:
                    print('launched exp ', slurm_cmd_script_path)
                else:
                    print('launch failed...')  
