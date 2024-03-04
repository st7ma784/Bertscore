from test_tube import HyperOptArgumentParser
import os

base= "/nobackup/projects/bdlan05/$USER/" if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk") else "$global_storage"
class baseparser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        self.add_argument("--dir",default=os.path.join(base,"data") ,type=str)
        self.add_argument("--log_path",default=os.path.join(base,"logs/",type=str)
        # self.opt_list("--learning_rate", default=0.0001, type=float, options=[2e-4,1e-4,5e-5,1e-5,4e-6], tunable=True)
        self.opt_list("--batch_size", default=180, type=int)
        
        #INSERT YOUR OWN PARAMETERS HERE
        from lsafunctions import get_all_LSA_fns
        lsakeys=list(get_all_LSA_fns().keys())
        self.opt_list("--LSAVersion",default="none",options=["none"]+lsakeys, tunable=True)
        self.opt_list("--all_layers", default=False, options=[False], tunable=True)
        self.opt_list("--perfect_match", default=False, options=[True,False], tunable=True)
        self.opt_list("--accelerator", default='auto', type=str, options=['auto'], tunable=True)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        #which model to use as inspired by the list on the bertscore github
        self.opt_list("--padding_length", default=128, type=int, options=[128,384], tunable=True)
        self.opt_list("--modelname", default="bert-base-uncased", type=str, options=["bert-base-uncased",
                                                                                    "roberta-base",
                                                                                    "xlm-roberta-base",
                                                                                    "distilbert-base-uncased",
                                                                                    "albert-base-v2",
                                                                                    #"xlnet-base-cased", #throws int to large error
                                                                                    # "xml-roberta-large",
                                                                                    #"t5-small", # needs to be passed decoder ids as input too **sigh 
                                                                                    "facebook/bart-base",], tunable=True)
                                                                                    
        #This is important when passing arguments as **config in launcher
        self.argNames=["dir","log_path","learning_rate","batch_size","modelname","precision","LSAVersion","accelerator","num_trials"]
    def __dict__(self):
        return {k:self.parse_args().__dict__[k] for k in self.argNames}

    # def parse_args(self):
    #     output=super().parse_args()
    #     print(output.__dir__)
    #     #<built-in method __dir__ of TTNamespace object at 0x7f9a0a846520> MB deduped)
    #     #   <class 'test_tube.argparse_hopt.TTNamespace'>

    #     return output
    
import wandb
from tqdm import tqdm


class parser(baseparser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False,**kwargs) # or random search
        self.run_configs=set()
        self.keys=set()
    def generate_wandb_trials(self,entity,project):
        wandb.login(key='9cf7e97e2460c18a89429deed624ec1cbfb537bc')
        api = wandb.Api()

        runs = api.runs(entity + "/" + project)
        print("checking prior runs")
        for run in tqdm(runs):
            config=run.config
            for key in config.keys():
                self.keys.add(key)
            #print(config)
            sortedkeys=list([str(i) for i in config.keys()])
            sortedkeys.sort()
            values=list([str(config[i]) for i in sortedkeys])
            code="_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        for trial in tqdm(trials):
            sortedkeys=list([str(i) for i in self.keys])
            sortedkeys.sort()
            values=list([str(trial.__dict__[k]) for k in sortedkeys if k in trial.__dict__])
            
            code="_".join(values)
            while code in self.run_configs:
                trial=hyperparams.generate_trials(1)[0]
                sortedkeys=list([str(i) for i in self.keys])
                sortedkeys.sort()
                values=list([str(trial.__dict__[k]) for k in sortedkeys if k in trial.__dict__])
                code="_".join(values)
        return trials
        
# Testing to check param outputs
if __name__== "__main__":
    
    #If you call this file directly, you'll see the default ARGS AND the trials that might be generated. 
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams.__dict__)
    for trial in hyperparams.generate_trials(10):
        print(trial)
        
