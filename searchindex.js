Search.setIndex({docnames:["api_doc/data","api_doc/envs","api_doc/eval","api_doc/index","api_doc/models","api_doc/policy","api_doc/simulators","api_doc/utils","faq/index","features/carla_benchmark","features/casezoo","features/datasets","features/index","features/policy_feature","features/simulator_feature","features/visualize","index","installation/index","model_zoo/cict","model_zoo/coil","model_zoo/implicit","model_zoo/index","model_zoo/lbc","tutorial/auto_run","tutorial/carla_tutorial","tutorial/core_concepts","tutorial/il_tutorial","tutorial/index","tutorial/rl_tutorial"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api_doc/data.rst","api_doc/envs.rst","api_doc/eval.rst","api_doc/index.rst","api_doc/models.rst","api_doc/policy.rst","api_doc/simulators.rst","api_doc/utils.rst","faq/index.rst","features/carla_benchmark.rst","features/casezoo.rst","features/datasets.rst","features/index.rst","features/policy_feature.rst","features/simulator_feature.rst","features/visualize.rst","index.rst","installation/index.rst","model_zoo/cict.rst","model_zoo/coil.rst","model_zoo/implicit.rst","model_zoo/index.rst","model_zoo/lbc.rst","tutorial/auto_run.rst","tutorial/carla_tutorial.rst","tutorial/core_concepts.rst","tutorial/il_tutorial.rst","tutorial/index.rst","tutorial/rl_tutorial.rst"],objects:{"core.data.carla_benchmark_collector":{CarlaBenchmarkCollector:[0,0,1,""]},"core.data.carla_benchmark_collector.CarlaBenchmarkCollector":{close:[0,1,1,""],collect:[0,1,1,""],reset:[0,1,1,""]},"core.data.dataset_saver":{BenchmarkDatasetSaver:[0,0,1,""]},"core.data.dataset_saver.BenchmarkDatasetSaver":{make_dataset_path:[0,1,1,""],make_index:[0,1,1,""],save_episodes_data:[0,1,1,""]},"core.envs":{BaseCarlaEnv:[1,0,1,""],BenchmarkEnvWrapper:[1,0,1,""],CarlaEnvWrapper:[1,0,1,""],ScenarioCarlaEnv:[1,0,1,""],SimpleCarlaEnv:[1,0,1,""]},"core.envs.BaseCarlaEnv":{close:[1,1,1,""],reset:[1,1,1,""],seed:[1,1,1,""],step:[1,1,1,""]},"core.envs.BenchmarkEnvWrapper":{reset:[1,1,1,""],step:[1,1,1,""]},"core.envs.CarlaEnvWrapper":{close:[1,1,1,""],info:[1,1,1,""],render:[1,1,1,""],reset:[1,1,1,""],seed:[1,1,1,""],step:[1,1,1,""]},"core.envs.ScenarioCarlaEnv":{close:[1,1,1,""],compute_reward:[1,1,1,""],get_observations:[1,1,1,""],is_failure:[1,1,1,""],is_success:[1,1,1,""],render:[1,1,1,""],reset:[1,1,1,""],seed:[1,1,1,""],step:[1,1,1,""]},"core.envs.SimpleCarlaEnv":{close:[1,1,1,""],compute_reward:[1,1,1,""],get_observations:[1,1,1,""],is_failure:[1,1,1,""],is_success:[1,1,1,""],render:[1,1,1,""],reset:[1,1,1,""],seed:[1,1,1,""],step:[1,1,1,""]},"core.eval":{CarlaBenchmarkEvaluator:[2,0,1,""],SingleCarlaEvaluator:[2,0,1,""]},"core.eval.CarlaBenchmarkEvaluator":{close:[2,1,1,""],eval:[2,1,1,""],reset:[2,1,1,""]},"core.eval.SingleCarlaEvaluator":{close:[2,1,1,""],eval:[2,1,1,""],should_eval:[2,1,1,""]},"core.models":{BEVSpeedConvEncoder:[4,0,1,""],BEVSpeedDeterminateNet:[4,0,1,""],BEVSpeedStochasticNet:[4,0,1,""],VehiclePIDController:[4,0,1,""]},"core.models.BEVSpeedConvEncoder":{forward:[4,1,1,""]},"core.models.BEVSpeedDeterminateNet":{forward:[4,1,1,""]},"core.models.BEVSpeedStochasticNet":{forward:[4,1,1,""]},"core.models.VehiclePIDController":{forward:[4,1,1,""]},"core.models.model_wrappers":{SteerNoiseWrapper:[4,0,1,""]},"core.models.model_wrappers.SteerNoiseWrapper":{forward:[4,1,1,""]},"core.policy":{AutoPolicy:[5,0,1,""]},"core.policy.AutoPolicy":{_forward_collect:[5,1,1,""],_forward_eval:[5,1,1,""],_reset_collect:[5,1,1,""],_reset_eval:[5,1,1,""]},"core.policy.base_carla_policy":{BaseCarlaPolicy:[5,0,1,""]},"core.simulators":{CarlaScenarioSimulator:[6,0,1,""],CarlaSimulator:[6,0,1,""]},"core.simulators.CarlaScenarioSimulator":{clean_up:[6,1,1,""],end_scenario:[6,1,1,""],get_criteria:[6,1,1,""],init:[6,1,1,""],run_step:[6,1,1,""]},"core.simulators.CarlaSimulator":{apply_control:[6,1,1,""],apply_planner:[6,1,1,""],clean_up:[6,1,1,""],get_information:[6,1,1,""],get_navigation:[6,1,1,""],get_sensor_data:[6,1,1,""],get_state:[6,1,1,""],init:[6,1,1,""],run_step:[6,1,1,""]},"core.simulators.base_simulator":{BaseSimulator:[6,0,1,""]},"core.simulators.base_simulator.BaseSimulator":{apply_control:[6,1,1,""],run_step:[6,1,1,""]},"core.utils.env_utils.stuck_detector":{StuckDetector:[7,0,1,""]},"core.utils.env_utils.stuck_detector.StuckDetector":{clear:[7,1,1,""],tick:[7,1,1,""]},"core.utils.others.visualizer":{Visualizer:[7,0,1,""]},"core.utils.others.visualizer.Visualizer":{done:[7,1,1,""],init:[7,1,1,""],paint:[7,1,1,""],run_visualize:[7,1,1,""]},"core.utils.planner.basic_planner":{BasicPlanner:[7,0,1,""]},"core.utils.planner.basic_planner.BasicPlanner":{clean_up:[7,1,1,""],get_incoming_waypoint_and_direction:[7,1,1,""],get_waypoints_list:[7,1,1,""],run_step:[7,1,1,""],set_destination:[7,1,1,""],set_route:[7,1,1,""]},"core.utils.planner.behavior_planner":{BehaviorPlanner:[7,0,1,""]},"core.utils.planner.behavior_planner.BehaviorPlanner":{run_step:[7,1,1,""]},"core.utils.simulator_utils.sensor_utils":{CollisionSensor:[7,0,1,""],SensorHelper:[7,0,1,""],TrafficLightHelper:[7,0,1,""]},"core.utils.simulator_utils.sensor_utils.CollisionSensor":{clear:[7,1,1,""]},"core.utils.simulator_utils.sensor_utils.SensorHelper":{clean_up:[7,1,1,""],get_sensors_data:[7,1,1,""],setup_sensors:[7,1,1,""]},"core.utils.simulator_utils.sensor_utils.TrafficLightHelper":{tick:[7,1,1,""]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"]},objtypes:{"0":"py:class","1":"py:method"},terms:{"0002":19,"00024":18,"0003":18,"003":4,"099":17,"100":[4,9,11,14,19],"1000":[14,18],"100000":18,"1000001":19,"100carla":20,"1109":18,"115":19,"120":19,"126":8,"128":[4,18,19],"160":14,"200":[7,18,19],"2000":[14,20,24],"200000":19,"2002":24,"2018":19,"2018end":19,"2019":[9,22],"2021":[16,18],"2080ti":26,"20ghz":28,"2477":18,"2484":18,"256":[4,18,19],"3000":18,"3061336":18,"320":14,"32g":[20,28],"360":18,"384":14,"400":[14,18],"44275":18,"48carla":20,"4gb":17,"500":[19,23],"5000":19,"512":[4,18,19],"52583":18,"600":[19,23],"60carla":20,"640":18,"65000":18,"75000":19,"800":[19,23],"8000":19,"8002":19,"8700":28,"9000":[1,6,18,19,23,24,26,28],"9002":[18,23,24,26,28],"9008":26,"9010":[18,19],"9016":28,"9050":6,"9361054":18,"9900k":26,"999":18,"abstract":[1,6],"case":[10,24],"class":[0,1,2,4,5,6,7,14],"default":[0,1,2,4,5,6,7,8,10,14,17,18,23,24,26,28],"export":19,"final":[1,2,4,7,14,22],"float":[1,2,4,6,7,11,14],"function":[0,13,18,25],"import":[10,17],"int":[0,1,2,4,5,6,7,11,14],"new":[1,5,8,10,25],"null":8,"return":[0,1,2,4,5,6,7,11,13,14],"short":[1,7,10],"static":6,"switch":[14,23,28],"true":[7,14,15,18,19,28],"try":16,"void":14,"while":[1,6,24],And:1,For:[0,2,5,9,11,13,14,15,24,26,28],Gas:19,Its:1,NOT:[1,6,24],Obs:11,One:[13,25],The:[0,1,2,4,5,6,7,8,9,10,11,14,15,16,18,19,20,22,23,24,25,26,28],Their:[5,11],Then:[4,5,14,15,17,22,23,24,26],There:[10,14],Will:14,_00000:11,_forward_collect:[5,13],_forward_ev:5,_interfac:5,_interface_xxx:5,_log:[18,26],_preload:[18,26],_reset_collect:5,_reset_ev:5,_variablefunctionsclass:4,abil:10,abl:[6,10,13,20],about:[1,6,14,19,20,28],abov:[4,15,26],academ:16,academia:16,acc:6,acceler:[11,14,18],acceleration_loss_weight:18,accord:[0,1,5,6,7,9,10,28],account:[7,14],accur:10,achiev:[6,20],across:16,act:13,action:[0,1,4,7,13,15,25],action_shap:4,actor:[1,4,6,7,14,24],actual:6,adapt:16,add:[1,4,5,6,7,8,10,11,14,15,28],added:[1,8,14,15],adding:8,addit:[0,2],afford:[9,21],after:[0,13,28],again:[0,8],agent:[7,14,15,17,20,25],agent_st:14,aggress:7,ahead:[7,14],ahenb:22,aim:[1,10],alexei:19,algorithm:28,alias:9,align:1,all:[0,1,2,5,6,7,9,10,11,13,14,16,19,23,24,25,28],allow:[9,10,11,13,24],along:1,alreadi:6,also:[1,6,7,9,10,14,17,18,24,25],alwai:1,amazonaw:17,among:14,amount:[9,10],anaconda:8,analyz:[7,25],angl:6,ani:[0,1,2,4,5,6,7,10,16,17,25],antonio:19,anyth:8,aonfigur:9,api:[8,9,14,16,24],aplli:6,appli:[4,6,7,16,26],applic:16,apply_control:6,apply_plann:6,apt:8,architectur:[19,24,26],arg:[1,4,6],args_later:4,args_longitudin:4,argument:[0,1,2,4,5,6,7,9,14],arrai:14,arrang:11,articl:18,asound:8,aspect:11,associ:14,async:24,asynchron:24,attach:7,attitud:1,aug:14,aug_cfg:7,augmant:7,augment:[7,14,19],author:[18,19,22],auto:[0,2,10,14,15,16,20,27],auto_pilot:14,auto_reset:[18,19],auto_run:[10,23],auto_run_cas:[10,23],autoeval_config:[18,19],autom:[18,19],automat:[0,7,11,14,24],autonom:[5,9,14,16,18,25,28],autopilot:[6,26],autoploit:19,autopolici:[3,23],autorun_config:23,avail:[0,2,14],averag:0,avoid:[6,7,25,28],baci:24,backbon:20,background:14,base:[1,5,6,7,16],base_carla_env:1,base_carla_polici:5,base_env:1,base_env_manag:[0,2],base_simul:6,basecarlaenv:3,basecarlapolici:3,baseenv:1,baseenvinfo:1,baseenvmanag:[0,2,13],baseenvtimestep:1,basesimul:3,bash:24,basic:[7,13,14,16,27],basic_plann:7,basicplann:[3,14],batch:13,batch_siz:[18,19],becaus:[8,24],been:19,befor:[6,14,18,24],begin:28,beginn:16,behavior:[1,7,10,13,18,19,24],behavior_plann:7,behaviorplann:[3,14],being:1,below:11,benchmark:[0,1,2,10,12,16,19,22,26],benchmarkdatasetsav:[3,11],benchmarkenvwrapp:[3,9],benckmark:20,besid:7,beta1:18,beta2:18,between:[4,14,15],bev:[4,6,14,19,23],bevspeedconvencod:3,bevspeeddeterminatenet:3,bevspeedstochasticnet:3,bin:[8,24],bird:[4,6,14,15,18,22,23,28],birdview:[4,14,19,23,28],block:14,booktitl:[19,22],bool:[1,2,4,6,7,14],both:[4,16,22],bradi:22,brake:[4,11,19,22,26],branch:19,branch_loss_weight:19,buffer:[7,25,28],build:[6,9,14,15,28],built:[10,24],cach:0,calcul:[1,7,23,25],call:[1,5,6,13,14,15,16,23,24],callabl:0,camera1_nam:11,camera2_nam:11,camera:[6,7,14,15,18,19,20,23,26],can:[0,1,2,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,23,24,25,26,28],canva:[1,7,15],captur:[19,20],car:7,card:8,carl:14,carla:[0,1,2,5,6,7,9,10,11,12,16,18,19,22,23,26,27,28],carla_099:17,carla_0:17,carla_ag:7,carla_benchmark_collector:0,carla_env:[10,24],carla_host:[18,19,23,26,28],carla_port:[18,19,23,26,28],carlabenchmarkcollector:[3,9],carlabenchmarkevalu:[3,9],carladataprovid:6,carlaenvwrapp:3,carlascenariosimul:[1,3],carlasim:24,carlasimul:[1,3,14],carlaue4:[17,20,24,26],casezoo:[12,16,22,27],categori:11,caus:6,cautiou:7,certain:[7,10,19,26],certainli:1,cfg:[0,1,2,5,6,7],chang:[0,1,4,5,6,7,14,16,18,19,23,24,26,28],changelanetown04:[9,20],channel:[6,14,18],chaotic:24,characterist:10,charactorist:10,cheat:21,check:[1,6,7,9,14,15,16,17,23],checkout:17,checkpoint:[18,19,26],chen2019lbc:22,chen:[18,22],choos:[1,9,14,18],cict:18,cict_datasets_train:18,cict_demo:18,cict_eval_gan:18,cict_eval_traj:18,cict_gan:18,cict_test:18,cict_train_gan:18,cict_train_traj:18,cict_traj:18,cil:[19,26],cils_datasets_train:[19,26],ckpt_path:[19,26,28],clean:[6,7],clean_up:[6,7,24],clear:[0,6,7,14],clear_up:7,client:[4,6,24],clip:4,clone:17,close:[0,1,2,18],closer:16,cloudi:14,code:[17,19,28],codevilla:19,coil:19,coil_data_collect:[19,26],coil_demo:[19,26],coil_ev:[19,26],coil_icra:[19,26],coil_train:[19,26],col_is_failur:[18,19],col_threshold:[7,14],collect:[0,5,11,13,16,18,20,25,27,28],collect_data:[18,20],collect_mod:[5,13],collector:[0,2,9,13,18,19,25],collet:18,collid:[6,7],collis:[6,7,10,14,25],collisionsensor:3,com:17,combin:[4,16],come:[6,10,14],command:[0,8,11,14,17,19,23,24,26],command_index:0,common:[6,10,12,17,18,19,24,26],commonli:[9,25],commun:24,compar:10,compil:17,complet:[16,28],complex:[13,16],compon:11,comput:[1,4,13,14],compute_reward:1,concat:[4,19],concept:[10,16,27],concern:28,conda:8,condit:[9,21,26],conf:8,confer:[19,22],config:[0,1,2,5,6,7,10,14,15,18,19,20,23,26,28],config_fil:10,configuarion:10,configur:[1,6,10,11,12,16,18,19,26],congratul:26,connect:[4,17,23,24],consist:[7,9,11,14,24],contain:[1,4,6,7,9,11,13,14,16,24,25,28],content:[11,14,28],continu:21,control:[1,4,5,6,10,13,14,16,18,19,22,24,25,26,28],conv:4,conveni:[9,10],convert:[1,6],convolut:[4,19],coordin:[4,14],copi:20,core:[0,1,2,4,5,6,7,10,13,16,27,28],corl:22,correct:5,correctli:13,correl:7,correspond:17,cost:26,could:[1,6,8],count:[0,6,7],cpu:28,creat:[1,6,7,10,11,14,15,16,26,27],criteria:[1,6,10],critic:4,crop:20,cross:7,csv:2,cuda:28,current:[1,2,4,6,7,10,14,24,28],current_loc:4,current_ori:4,current_spe:4,currrent:14,custom:[11,14,15,16,19,26],cutin:10,cutin_1:10,dai:20,data:[1,3,4,5,6,7,9,10,12,13,14,15,16,17,20,23,25,26],data_dict:7,data_id:5,dataset:[0,9,10,12,16,18,19,20,25,27],dataset_dir:20,dataset_metadata:11,dataset_metainfo:0,dataset_nam:11,dataset_path:[18,19,26],dataset_sav:0,datasets_train:[19,26],datasets_train_collector:19,deal:17,debug:[7,14],decis:[7,10,16,25],decompos:[16,25],deep:[10,16,25],defin:[0,1,5,7,9,10,11,13,14,16,25,28],defini:13,definit:[1,25],delet:[1,6],deliv:1,delta_second:14,demo:[10,18,19,20,23,26,28],depend:4,deploi:[9,16],deploy:[1,28],depth:14,derect:6,deriv:1,describ:11,descript:[6,16,23],design:[5,10,13,16,23],desir:11,dest:18,destin:[7,18],destroi:[6,7,24],detail:[1,9,10,14,16,18,19,20,28],detect:7,detector:7,determin:4,dev:8,develop:[16,17],devid:10,dian:22,dict:[0,1,2,4,5,6,7,13,14,15,18,19,23,26,28],dictanc:[6,14],dictionari:[4,14],differ:[5,7,10,11,13,14,25],differenti:4,difficulti:[10,16],dimens:4,ding:[0,1,2],dir:20,dir_path:[18,19],direct:[1,7,14],directli:[7,8,13,28],directori:[11,17,28],disabl:14,disable_two_wheel:[6,14,18,19],discrimin:18,displai:25,dist:17,distanc:[6,7,14],distribut:17,dive:16,divid:11,doc:[9,14,16,24,28],docker2:24,docker:[16,27],document:14,doi:18,done:[1,7,13],dongkun:18,dosovitskii:19,down_channel:18,down_dropout:18,down_norm:18,download:[16,20],dqn:[9,28],dqn_eval:28,dqn_main:28,draw:14,drive:[0,5,9,10,11,13,14,15,18,19,21,22,23,24,25,26,27],drive_len:4,dropout:[18,19],dure:[6,7,10,13,14],dut:6,each:[0,1,2,4,5,6,7,9,10,11,13,14,18,25],eas:16,easi:28,easier:24,easili:[7,10,11,16],easy_instal:17,ect:6,edu:8,effect:[6,11],effici:16,egg:17,ego:[7,10],elem:0,element:[7,14],els:25,embed:4,embedding_s:4,enabl:26,enable_field:5,encod:[4,7,20],encoder_embedding_s:4,encoder_hidden_dim_list:4,encount:6,end:[1,6,7,9,10,11,14,19,21,23],end_dist:6,end_episod:18,end_idx:6,end_loc:7,end_scenario:6,end_timeout:6,engin:[1,2,5,9,13,16,25,27],entangl:24,entir:[6,9,25],entri:[22,28],env:[0,2,3,5,8,10,16,18,19,20,23,25,26,28],env_cfg:[10,24],env_id:13,env_manag:[0,2,13,18,19],env_num:[18,19,20,28],env_param:0,env_util:7,env_wrapp:[18,19],envalu:2,envirion:1,environ:[0,1,2,5,6,7,8,9,10,13,15,16,18,20,22,25,28],envmanag:[0,2,5,9,13],episod:[0,1,2,5,6,9,10,12,14,18],episode_00000:11,episode_00001:11,episode_count:0,episode_metadata:11,episode_num:[18,19],episode_per_suit:2,episodes_data:0,episodes_per_suit:[18,19],equal:[5,13],error:[6,17],essenti:[6,10],establish:[6,10,24],etc:[8,13,23,25],eval:[3,5,9,10,13,18,19,20,25],eval_config:[18,28],eval_mod:13,evalu:[1,2,12,13,16,17,25,27],evalut:20,even:[10,16,24],event:[1,7],everi:[0,2,7,15],everyth:[17,23],eviron:[7,10],exactli:10,exampl:[5,9,10,13,14,15],exe:17,execut:4,exist:[6,10,16],exp:[18,19,26],expect:14,expert:[5,26],explain:[11,14],extend:20,extern:13,eye:[4,6,14,15,22,23,28],fail:[1,10],failu:10,failur:[1,2,10],fals:[4,7,14,18,19],faq:[16,17],far:20,farthest:7,fast:24,featur:[4,10],feel:16,felip:19,file:[0,1,2,7,8,9,10,11,15,16,23,26],final_channel:18,find:[0,2,6,8,9,14,24],finish:[10,26],first:[1,2,16,17,18,20,23,26],fix:[8,10,11,24],flexibl:16,flow:25,folder:[0,8,11,15,18,19,20,26],follow:[1,4,5,8,10,11,13,14,16,17,20,23,24,25,28],follw:11,forg:8,form:[0,2,5,10,11,25],format:[1,11],former:10,forward:[0,2,4,5,13,14],found:[8,15,19,28],fov:14,fps:[14,26],frame:[0,1,4,6,11,14,15,20],frame_skip:[15,28],framework:25,free:[6,16,21,24],freeli:11,freez:20,frequenc:2,frequent:[4,6],from:[0,1,2,4,6,7,8,9,10,11,13,14,17,21,25,28],front:[7,19,20,26],front_rgb:14,full:[9,19],fulli:4,fulltown01:[9,18,19],fulltown02:[9,19,28],fulltown04:9,fulltown:20,gail:25,gan:18,gan_ckpt_path:18,gan_loss_funct:18,gauss:4,geforc:26,gener:[0,4,6,12,14,18,25],geo:8,geos_c:8,get:[0,1,4,5,6,7,8,10,14,15,16,22,25,26,28],get_criteria:6,get_incoming_waypoint_and_direct:7,get_inform:6,get_navig:6,get_observ:1,get_sensor_data:6,get_sensors_data:7,get_stat:6,get_train_sampl:5,get_waypoints_list:7,gif:[1,7,15,28],git:17,github:17,give:19,given:4,global:[7,14,24],goal:[1,6],gohlk:8,gpu1060:28,gpu:[17,18,19,20,24],gpu_id:24,graph:13,greatli:16,green:14,ground:22,groundtruth:18,guid:[12,17,24],guidanc:24,guidenc:16,gym:[1,10,16,25],half:4,hand:[9,16,24],handl:[10,13],handler:7,happen:[1,6,8],hard:[10,14],hardwar:[17,26],has:[1,5,7,14,19,24],have:[0,2,5,7,8,11,13,14,16,17,25,26,28],hawk:9,head:20,head_embedding_dim:4,help:25,here:[9,14,15,26,28],hero:[1,4,6,7,10,14,25],hero_play:[1,6],hero_vehicl:7,hidden:4,hidden_dim:18,hidden_dim_list:4,hierach:18,hierarch:18,high:16,histor:17,histori:7,home:16,host:[1,6,10,17,23,24,26,28],hostport:26,hour:[20,26,28],how:[15,20,26,28],howev:10,http:[8,17,20],icra:19,ieee:[18,19],ignor:[1,8,9,28],imag:[4,6,7,11,12,14,15,19,22,23,24,25,28],image_cut:19,img_height:18,img_step:18,img_width:18,imit:[9,16,18,22,25,27],immedi:25,implicit:[9,21],impuls:7,includ:[6,10,14,15,16,19,22,25,26],inclut:6,incom:[1,7],index:[0,6,11,23],indic:16,individu:[10,24],indivisu:10,industri:16,info:[1,6,7,14,15],infom:0,inform:[1,6,7,9,11,12,13,15,19,20,25,28],inherit:1,init:[1,5,6,7,10,13,14],init_w:4,initi:[1,7],initla:7,inproceed:[19,22],input:[4,5,6,7,13,16,18,19,22,26,28],input_dim:18,instal:[16,24],instanc:[0,1,2,4,6,10,13,14,16,20],instant:25,instantan:7,instruct:[10,14,23],integr:[4,16],intel:28,intellig:[10,16],intens:14,intent:21,interac:10,interact:[0,1,2,5,14,25],interfac:[0,1,2,4,5,6,7,10,13,14,16],intern:[13,16,19],intial:6,introduc:[10,19],invok:4,involv:25,irl:25,is_crit:4,is_failur:1,is_junct:14,is_success:1,item:15,iter:[2,25],its:[0,5,6,7,9,11,14,17,24],itself:10,jingk:18,join:19,journal:18,jpeg:8,json:[0,10,11,23],judg:[2,7],judgement:[1,25],junction:[6,14],just:[8,13,17,24,26],k_d:4,k_i:4,k_p:4,keep:[7,13],kei:[0,5,10,11,14,15,23],kernel:4,kernel_s:[4,18],kind:[5,7,9,13,16,25],koltun:[19,22],kwarg:[1,4,6],label:[13,19,20],lane:[7,9,14,20,25],larg:28,last:[1,7,11,16,26],later:4,laterli:25,latter:10,layer:4,lbc:22,lbc_bev_ev:22,lbc_bev_test:22,lbc_image_ev:22,lbc_image_test:22,leanr:20,learn:[1,5,9,10,13,16,17,18,25,27],learn_mod:13,learner:[13,25],learning_r:[18,19],learning_rate_decay_interv:19,learning_rate_decay_level:19,learning_rate_threshold:19,left:14,legal:1,len_thresh:7,length:[1,4,7],letter:18,level:4,lfd:8,libcarla:[6,7],libgeo:8,libpng16:8,librari:8,licens:16,lidar:[6,14,18],lidar_nam:11,light:[1,6,7,10,14,15,28],lightweight:16,like:[7,9,10,11,13,19,23,26],limit:[7,14],line:[7,8,14,17],lingang:16,link:[6,8,24],linux:17,list:[0,4,5,6,7,9,14,15,20,22,24],literatur:[9,10,16],lmdb:[0,11],load:[1,6,11],load_state_dict:13,local:[6,7,14,17],localhost:[1,6,18,19,23,26,28],locat:[4,6,7,10,11,14],log:[1,4,8,20,28],log_dir:20,log_std_max:4,log_std_min:4,logdir:28,logic:[10,13],longitudin:4,look:[11,17],loop:18,lopez:19,loss:[13,18,26],loss_funct:19,low:[4,16],lower_fov:[14,18],lra:18,lsof:17,machin:26,mai:[1,5,6,7,8,10,11,13,14,17,18,23,28],main:1,mainli:[2,16,24],make:[0,1,5,6,8,9,10,11,13,14,16,20,24,28],make_dataset_path:0,make_index:0,malici:10,manag:[0,1,2,6,24],mani:20,manual:[8,10,24],map:[6,9,10,11,13,14,18,20,23,24,25],matthia:19,max:4,max_brak:4,max_ckpt_save_num:18,max_dist:18,max_steer:4,max_t:18,max_throttl:4,mean:[4,11,28],meanwhil:10,measur:[0,6,12,19],measurements_00000:11,meet:17,memori:[17,28],messag:6,met:14,meta:0,metadata:12,metainfo:0,method:[0,1,2,5,6,7,9,10,13,14,15,16,18,19,22,23,24,25],metric:12,mid:14,miiller:19,mimic:[19,26],min:4,min_dist:[7,14],minut:11,mkdir:[17,20],mode:[1,5,13,14,24,25],model:[3,5,13,16,18,25,27],model_configur:[18,19],model_path:20,model_rl:20,model_supervis:20,model_typ:[18,19],model_wrapp:4,models_town01:20,models_town04:20,modif:6,modifi:[14,16,17,18,19,20,25,26,28],modul:[8,9,13,14,16,25],modular:16,monitor:28,more:[7,10,13,14,15,19,20,28],most:[6,25],mostli:10,mount:6,mse:18,much:24,multi:[6,9,20,26],multimod:9,multipl:[20,24],must:[0,2,4,6,18],my_polici:13,mypolici:13,n_episod:0,n_epoch:18,n_pedestrian:[6,14],n_sampl:13,n_vehicl:[6,14],name:[6,7,10,11,14,18,19,23],namedtupl:1,natur:13,navig:[1,6,7,10,12,19,20,23,25,26,28],ndarrai:[1,14],nearbi:[7,14],necessari:11,need:[1,7,8,10,14,16,17,18,20,23,28],network:[4,13,18,19,20,25],neural:[4,13,19,25],neuron:19,newest:7,next:[6,7,14,24,26],nice:28,no_rend:14,node:[7,14],node_forward:14,nois:[4,5,18,19],noise_kwarg:4,noise_len:4,noise_rang:4,noise_typ:4,none:[0,1,2,4,5,6,7,14,15,18,19],noon:14,norm:18,normal:26,note:[0,1,2,17,18,26,28],noth:1,now:[6,17,26],npc:[6,9,10,14],npy:26,npy_prefix:18,num:[0,7,9,14,18,26,28],num_branch:18,num_class:19,number:[7,9,11,15,18,24],number_frames_fus:19,number_images_sequ:19,number_iter:19,number_of_branch:19,number_of_hour:19,number_of_loading_work:[18,19],nvidia:[17,24],obei:10,oberv:14,object:11,obs:[1,4,10,13,14,15,18,19,23],obs_cfg:[0,7],obs_shap:4,observ:[0,1,4,5,6,7,11,13,14,15,23,25],obtain:[1,7,18],occupi:24,off:[1,6,14,28],off_road:6,offici:17,offscreen:20,often:8,old:[6,8],ollid:1,onc:[0,2,6,7,9,26],one:[1,2,4,6,7,8,9,11,13,14,15,20,24],ones:7,onli:[0,1,2,5,10,14,16,24],onlin:15,onto:15,open:16,opendilab:[16,17,20],opengl:20,oper:[5,10,25],option:[0,1,2,4,5,6,7,13,14,15,17],order:[1,9,10,11,13,14,25],org:20,organ:11,orient:[4,6,11,14],oserror:8,other:[0,1,5,6,7,9,10,12,13,14,16,25,28],otherwis:20,out:1,out_dim:18,output:[4,13,15,16,18,23,25,26,28],overtak:7,overview:12,own:[13,14,17],pack:17,packag:17,pad:18,page:[16,17,18,19,20,24],paint:[7,14,15],pair:7,paper:19,parallel:[9,13],param:[0,1,2,9],paramet:[1,2,6,10,24,25,28],paremet:24,parent_actor:7,pars:[9,10],parse_routes_fil:10,parse_scenario_configur:10,part:[16,23,25,26,28],partit:11,pass:[0,9,10,13],path:[0,6,8,15,18,20,26,28],path_to_your_checkpoint:26,path_to_your_dataset:26,pcm:8,pedestrian:[6,9,11,14],pend:22,per:28,percept:[19,25],perform:[4,10,12,20,22,26],period:6,philipp:22,physic:24,pictur:28,pid:[4,5,13,22],pilot:14,pip:[8,17],pipelin:19,pixel_loss_funct:18,pixel_loss_weight:18,pixels_ahead_vehicl:14,pixels_per_met:[14,19,23],placement:18,plan:[16,25],planner:[6,7,12,18,19,25],planner_dict:14,platform:[1,16],player:6,pleas:26,plug:8,png:[8,11],point:[6,10],points_per_second:[14,18],polici:[0,1,2,3,9,10,12,16,18,19,20,22,25,26,27,28],policy_config:[18,19],policy_hideen_s:4,policy_kwarg:[0,2],polymorph:[13,16],port:[1,6,10,17,18,20,23,24,26,28],posit:[7,10,14,18,19,23],position_rang:14,possess:10,possibl:24,post:[0,18],post_process_fn:0,postpocess:11,potenti:18,power:13,pre:[18,20,28],pre_train:[18,19],pred_len:18,pred_t:18,predict:[18,19,22,26],prefix:18,preload_model_alia:[18,19],preload_model_batch:[18,19],preload_model_checkpoint:[18,19],prepar:20,prerequisit:[16,27],pretrain:22,print:[1,7,14,26],probabl:28,problem:17,procedur:[13,25],proceed:14,process:[0,11,16,18,19,22,26,27,28],process_transit:5,promot:25,properti:[0,1,2,6,13],proport:4,prosedur:11,protenti:18,provid:[0,1,2,5,6,7,9,11,14,16,20,22,23,24],pth:[18,19,20],publish:8,pull:24,put:[13,15,20],py3:17,pyenv:8,pypi:17,python:[8,10,14,16,18,20,22,23,24,26,28],pythonapi:17,pythonlib:8,pytorch:[16,17],queue:[2,7],quick:[16,19,24],quickli:[9,10,16,23,27],rain:14,raini:14,rais:[10,17],ran:[6,14],ran_light:6,random:[1,6,11,14],randomli:[1,10,14],rang:[4,7,14,18],rate:[9,10,19,26],reach:4,ready_ob:13,real:[4,10,14,16,26],real_brak:11,real_steer:11,real_throttl:11,realiz:[13,16],reason:24,receiv:7,recommend:[16,24],record:[1,6,7,14,15,17],red:[6,7,14],redesign:10,reduc:16,refer:[6,10,12,26],reflect:10,regardless:24,regist:7,reinforc:[1,5,13,16,17,25,27],reinstal:8,relat:[1,5,6,17],releas:[1,7,17,24],remain:[6,14],remov:[7,18,19],render:[1,2,7,14,15,25],renfirc:16,repeat:[4,10],replac:6,replai:[25,28],repositori:17,repres:[9,10,11,25],requir:[7,17],res:19,research:26,reset:[0,1,2,5,6,9,10,13,23,24],reset_param:2,resnet34:19,resolut:[14,18,19],resourc:1,respect:1,result:[1,2,13,14,15],retriv:1,review:25,reward:[1,2,10,15,23,25],rgb:[14,15,18,19,20,22,23,26],right:14,road:[1,6,7,9,10,14,15,16],roadopt:7,robot:[18,19,22],roll:25,rong:18,root:20,rotat:[14,18,19,23],rotation_frequ:[14,18],rotation_rang:14,rout:[1,6,7,9,10,14,20,23,25],route_fil:10,route_pars:10,routepars:10,router:7,rpc:17,rtx:26,run:[0,1,2,4,5,6,7,8,9,12,13,15,16,18,22,24,25,26,27,28],run_step:[6,7],run_visu:7,runner:10,runnign:6,runtim:[1,2,24],safe:7,safeti:7,same:[1,5,6,7,10,11,13,14,20,24,26,28],sampl:[0,12,13,18,19,23,25,26,28],save:[0,1,6,7,10,11,15,20,26,28],save_dir:[0,15,18,20,26,28],save_episodes_data:0,save_interv:18,save_schedul:19,saver:[0,26],scalar:4,scenario:[1,6,10,16,27],scenario_fil:10,scenario_manag:6,scenario_nam:10,scenario_pars:10,scenariocarlaenv:[3,10,23],scenarioconfigurationpars:10,scenarioenv:10,scenariosimul:10,scneario:6,screen:[1,7,15,23,26,28],scrip:26,script:17,sdl_hint_cuda_devic:20,sdl_videodriv:20,search:7,second:[4,14,18],section:11,see:[17,20,23,26,28],seed:[1,9,11],segment:14,select:[10,14,19,25],semant:4,send:[6,24],sens:[8,14],senser:11,sensor:[1,6,7,12,15,18,19,23],sensor_data:0,sensor_util:7,sensorhelp:3,sent:1,server:[1,6,14,16,18,19,23,26,27,28],set:[1,2,4,5,6,7,9,10,13,14,15,16,18,19,20,24,28],set_destin:7,set_rout:7,setup_sensor:7,setuptool:8,sevar:10,sever:[0,2,5,6,9,10,13,14,24],shanghai:16,shape:4,share:13,shared_memori:[18,19],should:[1,2,5,11,13,20,26],should_ev:2,show:[7,8,9,10,14,15,23,24,26,28],show_text:15,shown:[10,11,14,20,23],side:[4,24],signal:[1,4,5,6,13,28],similarli:25,simpl:[1,9,10,16,22,27],simple_rl:28,simplecarlaenv:[3,11,23,24],simpli:[1,8,14,16,17,23,28],simplifi:16,simualt:16,simualtor:[6,7,14,24],simul:[0,1,3,7,8,9,10,11,12,15,16,17,18,19,23,24,25],simulator_util:7,simultan:24,singal:[19,22,26],singl:[1,2,6,9,10,20,23,26],singlecarlaevalu:3,situat:15,size:[4,14,18,19,23,28],skip:15,sky:20,slave:8,small:28,soft:14,solv:8,some:[1,5,6,7,9,10,13,14,15,17,20,24,25],someth:20,sound:8,sourc:[0,1,2,4,5,6,7],sourec:16,space:1,spawn:[11,14],special:[0,5],specif:[1,5,10],specifi:[8,10,14,24],speed:[4,6,7,11,14,19,26,28],speed_branch:19,speed_factor:[18,19],speed_limit:14,speed_modul:19,speed_thresh:7,srunner:10,stage:[18,20,22],standard:[0,1,5,9,10,14,16,19,25,28],start:[0,1,5,6,7,9,10,11,14,16,19,20,26,27],start_episod:[0,18],start_loc:7,state:[1,6,7,14,15],state_dict:13,statu:[1,2,6,10,12,25],std:4,steer:[4,5,11,19,22,26],steernoisewrapp:3,step:[1,4,6,7,11,14,15,19,24,26],stochasticli:4,storag:26,store:[0,1,2,5,6,7,10,11,13,14,25],str:[0,1,4,5,6,7],straight:[9,14],straightli:1,straighttown01:20,stride:[4,18],structur:[1,12],stuck:[1,7],stuck_detector:7,stuck_is_failur:[18,19],stuckdetector:3,sub:14,subprocess:20,succe:[1,10],success:[1,2,9,10,25],successfulli:[1,17],sucha:6,sudo:8,suggest:[11,14],suit:[0,1,2,9,10,11,13,18,19,20,28],suitabl:[10,13,19,25],sunset:14,supervis:[5,13,20],supervised_model:20,supervised_model_path:20,support:[5,14,16,18,19,25,26],suppos:1,sure:[0,6,24],surround:[6,7],sync:[14,24],sync_mod:14,synchron:[7,14,24],system:[6,8,17,24,28],tabl:[22,26],tag:[11,14,15,17],tailgat:7,take:[4,6,7,14,15,19,20,22,26,28],taken:7,tar:[17,20],target:[1,4,5,6,7,9,14,16,19,22,27],target_forward:14,target_loc:4,target_spe:[4,18,19],task:[1,16],tcp:[1,6,24],teach:22,tensor:4,tensorboard:28,term:4,termin:28,test:[9,10,13,16,20],text:15,than:14,thecarla:6,thei:[9,10,14],them:[0,2,7,8,11,23],thi:[1,6,7,8,10,13,14,15,16,17,19,20,23,24,26],thread:26,three:[9,10,13,14,26],thresh:7,threshold:[7,14],throttl:[4,11,19,22,26],through:[14,23,24],throughout:14,thu:25,thw:4,tick:[1,6,7,11,14,24],time:[1,4,6,7,13,14,20,24,26],timeout:[6,7],timestamp:[11,14],titl:[18,19,22],tl_di:[11,14],tl_state:[11,14],tm_port:[1,6,24],togeth:[6,10,13,18,22,25,28],too:28,tool:[10,16],toolkit:10,top_rgb:14,torch:4,total:[1,9,11,14,18],total_diat:6,total_light:14,total_lights_ran:14,town01:[9,14],town02:[9,20],town03:10,town04:10,town05:10,town1:9,town2:[9,20],town:[6,9,10,11,14,18,20],town_nam:6,trace:[6,7],track:[7,14],traffic:[1,6,7,10,14,15,24,28],trafficlighthelp:3,train:[2,9,10,13,16,17,24,25,27],train_config:[18,19,28],train_data:13,train_dataset_nam:[18,19],train_host_port:20,train_it:2,train_rl:20,train_sl:20,traj_ckpt_path:18,traj_model:18,trajectori:[0,21],tree:1,trigger:[1,10],truth:22,tune:28,tupl:[1,4,7],turn:[9,14],tutori:[16,19,26],two:[4,11,14,15,18,20,22,23,25],txt:[0,9],type:[4,8,10,11,14,15,18,19,23,28],ubuntu:[8,17,28],uci:8,uhl:22,uncontrol:10,under:[13,16,17,19,26],unifi:[11,14,16],uniform:4,union:[0,4],unless:2,unpair:18,unreach:7,unsuccessfulli:1,up_channel:18,up_dropout:18,up_norm:18,updat:[5,6,7,16,24,25],upper_fov:[14,18],urban:[9,21],usag:[10,16],use:[1,4,5,6,8,10,13,14,15,16,17,19,20,24,25,26,28],used:[0,1,2,4,6,7,8,9,10,11,14,16,18,19,23,24,25],user:[0,2,5,7,9,10,11,13,14,15,16,17,24],uses:[0,1,2,5,6,7,10,13,14,16,17,18,24,28],using:[16,18,21,22,25,27],usual:[8,13,24,26,28],util:[1,3],v100:20,val_host_port:20,valu:[1,2,4,5,7,14],vari:[25,28],variable_weight:19,varieti:16,variou:[10,13],vector:[4,14],vehicl:[1,4,5,6,7,9,10,11,14,16,19,24,25,26],vehiclepidcontrol:3,veloc:18,velocity_loss_weight:18,verbos:[14,18,19],veri:[10,28],version:[6,8,9,17,25],via:[16,18,19,27],video:[1,7,15,18,28],view:[4,6,14,15,18,22,23,28],vis:18,visual:[1,2,3,12,16,18,19,22,25,26,27,28],visualizas:15,vladlen:[19,22],volum:18,vsiualiz:28,wai:[5,7,13,16,20,28],wait:24,walker:[6,7,10,14],wang:18,want:[17,20,23,26,28],wapoint:7,watch:15,waypoint:[1,4,5,6,7,9,11,14,16,22,27],waypoint_list:14,waypoint_num:[7,14,18,19],weather:[6,9,10,11,14,24],weight:[4,18,20,22,28],weight_decai:18,well:[5,6,10,16,20,23,25],west:17,wet:14,wget:[17,20],what:[10,13],whatev:23,wheel:[8,14],when:[0,1,5,6,7,10,13,14,15,23,24,26,28],whether:[1,2,4,6,7,14,15,17],which:[1,4,5,6,9,10,13,14,18,22,23,24,25],wide:[9,19],window:[7,8,17],winerror:8,within:[4,7],without:[0,2,10,16],work:[1,5,13,23,28],worker:13,world:[6,7,14,17,20,24,25,26],wrap:[1,4,9],wrapper:[1,4],write:0,written:28,writter:7,wrong:[1,6],wrong_direct:6,www:8,x86_64:17,xiao:9,xiong:18,xml:[10,23],xvf:20,xvzf:17,xxx:10,xxx_mode:[5,13],year:[18,19,22],yellow:14,you:[6,8,9,10,14,15,16,17,18,19,20,22,23,26,28],your:[8,14,20,23,24,26,28],yourself:8,yue:18,yuehua:18,yunkai:18,zexi:18,zhang:18,zhou:22,zoo:16},titles:["data","envs","eval","API Doc","models","policy","simulators","utils","FAQ","Benchmark Evaluation","Casezoo Evaluation","Datasets","Features","Policy Features","Simulator Features","Visualization","DI-drive Documentation","Installation","from Continuous Intention to Continuous Trajectory","Conditional Imitation Learning","End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances","Model Zoo","Learning by Cheating","Auto policy running and visualization","Carla tutorial","Core Concepts and Processes","Simple Imitation Learning","Tutorial","Simple Reinforcement Learning"],titleterms:{"import":8,With:20,afford:20,alsa:8,api:[3,17],auto:23,autopolici:5,basecarlaenv:1,basecarlapolici:5,basesimul:6,basic:24,basicplann:7,behaviorplann:7,benchmark:[9,11,18,20],benchmarkdatasetsav:0,benchmarkenvwrapp:1,bevspeedconvencod:4,bevspeeddeterminatenet:4,bevspeedstochasticnet:4,can:8,carla:[8,14,17,20,24],carlabenchmarkcollector:0,carlabenchmarkevalu:2,carlaenvwrapp:1,carlascenariosimul:6,carlasimul:6,casezoo:[10,23],cheat:22,collect:[19,22,26],collisionsensor:7,common:14,concept:[24,25],condit:19,configur:14,confxxx:8,content:16,continu:18,core:25,cost:20,creat:24,data:[0,11,18,19],dataset:[11,22,26],displai:20,doc:3,docker:24,document:16,download:17,drive:[16,17,20,28],easy_instal:8,egg:8,end:20,engin:[17,28],env:1,episod:11,error:8,eval:2,evalu:[9,10,18,19,20,22,26,28],faq:8,featur:[12,13,14,16],free:20,from:18,gener:11,get:17,guid:10,imag:9,imit:[19,21,26],implicit:20,inform:14,instal:[8,17],intent:18,learn:[19,20,21,22,26,28],lib:8,libjpeg:8,libpng:8,main:16,measur:11,metadata:11,method:21,metric:9,model:[4,19,20,21,22,26,28],navig:14,other:[11,21],overview:[10,14],perform:9,planner:14,polici:[5,13,23],prepar:18,prerequisit:[17,28],problem:8,process:25,python:17,quickli:24,refer:9,reinforc:[20,21,28],result:[18,20],run:[10,14,17,20,23],sampl:9,scenario:23,scenariocarlaenv:1,sensor:[11,14],sensorhelp:7,server:[17,20,24],shape:8,simpl:[26,28],simplecarlaenv:1,simul:[6,14],singlecarlaevalu:2,start:23,statu:14,steernoisewrapp:4,structur:11,stuckdetector:7,tabl:16,target:23,test:[18,22],town01:20,town04:20,trafficlighthelp:7,train:[18,19,20,22,26,28],trajectori:18,tutori:[24,27],urban:20,using:[20,28],util:7,vehiclepidcontrol:4,via:24,visual:[7,15,23],waypoint:23,without:20,zoo:21}})