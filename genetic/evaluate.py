from utils import Utils, GPUTools
import importlib
from multiprocessing import Process
import time, os, sys
from asyncio.tasks import sleep
import cnn
import utils

class FitnessEvaluate(object):

    def __init__(self, individuals, log, pop_size ):
        self.individuals = individuals
        self.log = log
        self.pop_size = pop_size

    def generate_to_python_file(self,gen_no):
        self.log.info('Begin to generate python files')
        for i in range(len(self.individuals)):
            # if self.individuals[i].change == 0:
            Utils.generate_pytorch_file(self.individuals[i])
        self.log.info('Finish the generation of python files')


    # def evaluate(self):
    def evaluate(self, min_complexity,gen_no):                                                                                 ##########
        """
        load fitness from cache file
        """
        self.log.info('Query fitness from cache')
        _map = Utils.load_cache_data()
        _count = 0
        # if gen_no <= 9:
        for indi in self.individuals:
            indi_no=indi.id[-2:]
            indi_id = int(indi_no)
            _key, _str = indi.uuid()
            if _key in _map and indi.change==0:
                _count += 1
                _acc = _map[_key][0]
                _complexity =int(_map[_key][1])
                if int(_complexity) < int(min_complexity):
                    _cv =1
                else:
                    _cv=0
                self.log.info('Hit the cache for %s, key:%s,acc:%.5f, min_complexity:%d, complexity:%d' % (indi.id, _key, float(_acc), min_complexity, _complexity))
                indi.acc = float(_acc)
                indi.complexity = int(_complexity)
                indi.cv = _cv

        self.log.info('Total hit %d individuals for fitness'%(_count))

        has_evaluated_offspring = False
        for indi in self.individuals:
            if indi.acc <= 0 or indi.change == 1 or (8> gen_no > 4 and 0<indi.acc<0.5)or (gen_no>7 and 0<indi.acc<0.7 and indi.cv==1):
                print("evaluate...")
                has_evaluated_offspring = True
                time.sleep(60)
                gpu_id = GPUTools.detect_available_gpu_id()
                while gpu_id is None or gpu_id == 3:
                    time.sleep(300)
                    gpu_id = GPUTools.detect_available_gpu_id()
                if gpu_id is not None and gpu_id != 3:
                        file_name = indi.id
                        _key, _str = indi.uuid()
                        self.log.info('Begin to train %s'%(file_name))
                        module_name = 'scripts.%s'%(file_name)
                        if module_name in sys.modules.keys():
                            self.log.info('Module:%s has been loaded, delete it'%(module_name))
                            del sys.modules[module_name]
                            _module = importlib.import_module('.', module_name)
                        else:
                            _module = importlib.import_module('.', module_name)
                        _class = getattr(_module, 'RunModel')
                        cls_obj = _class()
                        net = cnn.EvoCNNModel(indi)
                        param = utils.StatusUpdateTool.get_total_params(net)
                        p = Process(target=cls_obj.do_work, args=('%d' % (gpu_id), file_name,_key, _str, gen_no,param))
                        p.start()

            else:

                file_name = indi.id

                self.log.info('%s has inherited the fitness as %.5f,complexity as %d, no need to evaluate' % (file_name, indi.acc, int(indi.complexity)))                   ##########
                if os.path.exists('./populations/after_%s.txt' % (file_name[4:6])):
                    print("file exist true")
                    f = open('./populations/after_%s.txt'%(file_name[4:6]), "r")
                    lines = f.readlines()
                    indi_exist = False
                    for line in lines:
                        if file_name == line[0:8]:
                            indi_exist = True

                    if indi_exist == False:
                        f_a = open('./populations/after_%s.txt'%(file_name[4:6]), 'a+')
                        f_a.write('%s,%.5f,%d\n' % (file_name, indi.acc, int(indi.complexity)))
                        f_a.flush()
                        f_a.close()
                    else:
                        f_w = open('./populations/after_%s.txt'%(file_name[4:6]), "w")
                        for line in lines:
                            if file_name == line[0:8]:
                                print("acc from cache:",line[0:8])
                                str = '%s,%.5f,%d\n' % (file_name, indi.acc, int(indi.complexity))
                                line = line.replace(line, str)
                            f_w.write(line)
                        f_w.flush()
                        f_w.close()
                    f.flush()
                    f.close()

                else:
                    f = open('./populations/after_%s.txt'%(file_name[4:6]), "a+")
                    f.write('%s,%.5f,%d\n' % (file_name, indi.acc, int(indi.complexity)))
                    f.flush()
                    f.close()

        """
        once the last individual has been pushed into the gpu, the code above will finish.
        so, a while-loop need to be insert here to check whether all GPU are available.
        Only all available are available, we can call "the evaluation for all individuals
        in this generation" has been finished.

        """
        if has_evaluated_offspring:

            all_finished = False
            while all_finished is not True:
                time.sleep(300)
                all_finished = GPUTools.all_gpu_available()
                gpu_id = GPUTools.get_available_gpu_ids()
                if gpu_id == ['0', '1', '2'] or gpu_id == ['0', '1', '2']:
                    all_finished = True
                print("available_gpu_id",gpu_id)


        """
        the reason that using "has_evaluated_offspring" is that:
        If all individuals are evaluated, there is no needed to wait for 300 seconds indicated in line#47
        """
        """
        When the codes run to here, it means all the individuals in this generation have been evaluated, then to save to the list with the key and value
        Before doing so, individuals that have been evaluated in this run should retrieval their fitness first.
        """


        if has_evaluated_offspring:
            file_name = './populations/after_%s.txt'%(self.individuals[0].id[4:6])
            assert os.path.exists(file_name) == True
            f = open(file_name, 'r')
            fitness_map = {}

            for line in f:                                                                            ############
                if len(line.strip()) > 0:
                    line = line.strip().split(',')
                    list = []
                    list.append(float(line[1]))
                    list.append(int(line[2]))
                    fitness_map[line[0]] = list

            f.close()


            for indi in self.individuals:
                if indi.change ==1 or indi.acc<=0 or gen_no > 4:
                    if indi.id not in fitness_map:
                        self.log.warn('The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 120 seconds'%(indi.id, file_name))
                        sleep(120) #
                    indi.acc = fitness_map[indi.id][0]
                    indi.complexity = int(fitness_map[indi.id][1])



        else:
            self.log.info('None offspring has been evaluated')


        # next_min_complexity = min_complexity
        num_fre = 0
        num_indi = 0
        infre_com_list =[]

        for indi in self.individuals:
            num_indi += 1
            indi.complexity = int(indi.complexity)
            if abs(indi.complexity - min_complexity) < 1/10*min_complexity:
                indi.cv = 1
                indi.mean = indi.acc
                num_fre += 1
            else:
                if indi.complexity > min_complexity:
                    indi.cv = 0
                    infre_com_list.append(abs(int(indi.complexity) - min_complexity))
                    # indi.mean = indi.acc - (float(num_fre) / num_indi) * (abs(float(int(indi.complexity) - min_complexity)) / min_complexity)
                else:
                    indi.cv = -1
                    infre_com_list.append(abs(int(indi.complexity) - min_complexity))
                    # indi.mean = indi.acc - (float(num_fre) / num_indi) * (abs(float(int(indi.complexity) - min_complexity)) / min_complexity)
                # infre_com_list.append(indi.complexity)


        for indi in self.individuals:
            if indi.cv == 1:
                indi.mean = indi.acc
            else:
                infre_com_list.sort()
                indi.mean = (num_fre / num_indi) * indi.acc - (1 - num_fre / num_indi) * (
                            abs(float(int(indi.complexity) - min_complexity)) / infre_com_list[-1])

        for indi in self.individuals:
            print('indi:%s, Acc:%.3f, Complexity:%d, Mean: %.3f,CV:%d,change:%d' % (
            indi.id, indi.acc, int(indi.complexity), indi.mean, indi.cv, indi.change))








