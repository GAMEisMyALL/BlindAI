# Instruction to run #
agent name: KiraAI, the `Main.py` is only used to show how to run this agent within the DareFightingICE game.The most important files I would like to submit are: KiraAI.py(in the folder "`source_code`"),`DQN_model.keras`(in the folder "`trained_model`") 

---
### Installation: ###
- Python 3.12 installed
- Use the requirements.txt: `pip install requirements.txt` **or** Create and activate conda env:
> conda env create -n fightingice -f environment.yaml  
> conda activate fightingice

- boot the `DareFightingICE-6.3.1` with the option `--grpc-auto` 
- run the file `Main.py` directly.(p.s. use `python Main.py` might cause `FileNotFoundError: [WinError 3]`




### File Description:###
- `train.py` is a file used to train TestAI and build the model `my_model.keras`(which is inside the folder "
`trained_model`".
- `train2.py` is a file used to train the KiraAI based on the model `my_model.keras`
- `Main.py` is the file to let two agents fight in grpc mode. 


### Structure: ###
    KirariAI/					  		# deprecated AI's name  
    │  
    ├── KiraAI/ 				  		# now my AI's name
    │   ├── Data/
    |   |   └──sounds/			  		# sample audio resources
	|	|                
    │   ├── logs/  				  		# temp
    │   ├── trained_model/
	|	|	├── DQN_model.keras			# DQN model
	|	|	├── DQN_model2.keras
	|	|	├── DQNsoundAdded_model.keras
	|	|	├── my_model.keras			# pre-trained model
	|	|	└── processed_data.npz		# label on audios
	|	|
    |   ├── with_randomAI/
	|	|	├── HPMode_KiraAI_KickAI_2024.07.05-17.33.07.csv
	|	|	└── ...						# files used to analyze fighting results
	|	|
    │   └── with_testAI/
	|		├── HPMode_KiraAI_Kirari_2024.07.06-11.55.04.csv
	|		└── ...						# files used to analyze fighting results
    │  
    ├── source_code/  					# all codes here
    │   ├── _pycache_/  
    │   ├── logs/  
    |   ├── src/						# same codes in Generative-Sound-AI-main/src
    |   |   ├── _pycache_/
    |   |   ├── character_audio_handler.py
    |   |   ├── config.py
    |   |   ├── constants.py
    |   |   ├── core.py
    |   |   └── utils.py
    |   |
    |   ├── calculate_win_radio.py		# similar codes in BlindAI/analyze_fight_result.py
    |   ├── data_processing.py			# to bulid the processed_data.npz
    |   ├── KiraAI.py					# the agent KiraAI(model DQN_model.keras)
    |   ├── Main.py						
    |   ├── randomAI.py					# ramdomly choose command to fight
    |   ├── testAI.py					# which the old name is KirariAI, the agent to test the KiraAI
    |   ├── train.py					# to build the pre-trained model my_model.keras
    │   └── train2.py 					# to build the model DQN_model.keras
    │  
    ├── environment.yaml  
    ├── requirements.txt     
    ├── README.md     
    └── Zifan_FightingICE2024.pptx 
 

### Contact me###
If there's illegal thing or problem inside the project, please e-mail me:  
     yezifan1218@outlook.com **or**  `yzifan1106@163.com`  

(It's okay for me to just brush me out if the agent can't work:(