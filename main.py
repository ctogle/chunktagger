#!/home/cogle/anaconda3/bin/python3.6
import pdb
import util,dataset,model,forest


def main():
    '''Main loop for use creates Fields using training/test data, instantiates 
    a model (loading existing parameters if specified), trains the model if 
    specified, and runs the wiki data work example if specified.'''
    config = util.gather()
    data = dataset.fields(config)
    if config.forest:return forest.newforest(config,data)
    else:return model.newmodel(config,data)


if __name__ == '__main__':main()
