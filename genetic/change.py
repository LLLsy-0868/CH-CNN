from utils import Utils, GPUTools
import random
import utils
import copy

import cnn
import numpy as np

class Change(object):

    def __init__(self, pops, log, pop_size ):
        self.pops = pops
        self.log = log
        self.pop_size = pop_size

    def generate_to_python_file(self,gen_no,new_indi):
        self.log.info('Begin to generate python files')
        for i in range(len(new_indi)):
            Utils.generate_pytorch_file(new_indi[i])


        self.log.info('Finish the change of python files')


    def change_indi(self, indi):

        new_indi = copy.deepcopy(indi)
        conv_index_list = []
        inidi_error = np.abs(indi.complexity - self.pops.min_complexity)

        for i, unit in enumerate(indi.units):
            if unit.type == 1:
                conv_index_list.append(i)
        upblock = []
        downblock = []
        for i in range(len(conv_index_list)):
            if indi.units[conv_index_list[i]].block_type == 0 or indi.units[conv_index_list[i]].block_type == 1 or indi.units[conv_index_list[i]].block_type == 3:
                upblock.append(conv_index_list[i])
            elif indi.units[conv_index_list[i]].block_type == 2 or indi.units[conv_index_list[i]].block_type == 4:
                downblock.append(conv_index_list[i])

        # if inidi_error < 0.4*self.pops.min_complexity:
        #     threshold = 0.2
        # elif inidi_error < 0.8*self.pops.min_complexity:
        #     threshold = 0.4
        # else:
        #     threshold = 0.6

        if indi.cv == 0 and len(upblock) > 0:
            for i in range(random.randint(0,len(upblock)-1)):
                change_num = random.randint(0, len(upblock) - 1)
                if new_indi.units[upblock[change_num]].block_type == 0 or new_indi.units[upblock[change_num]].block_type == 1:
                        f = random.random()
                        if f < 0.3:
                            new_indi.units[upblock[change_num]].block_type = 2
                        elif f < 0.6:
                            new_indi.units[upblock[change_num]].block_type = 3
                        else:
                            new_indi.units[upblock[change_num]].block_type = 4
                elif new_indi.units[upblock[change_num]].block_type == 3:
                        f = random.random()
                        if f < 0.4:
                            new_indi.units[upblock[change_num]].block_type = 4

        elif indi.cv == -1:
            if len(downblock) > 0:
                for i in range(random.randint(0,len(downblock)-1)):
                    change_num = random.randint(0, len(downblock) - 1)
                    f = random.random()
                    if  new_indi.units[downblock[change_num]].block_type == 2:
                        if f < 0.2:
                            new_indi.units[downblock[change_num]].block_type = 0
                        elif f < 0.5:
                            new_indi.units[downblock[change_num]].block_type = 1
                    elif new_indi.units[downblock[change_num]].block_type == 4:
                        if f < 0.2:
                            new_indi.units[downblock[change_num]].block_type = 3
                        elif f < 0.5:
                            new_indi.units[downblock[change_num]].block_type = 2
            if len(upblock) > 0:
                for i in range(random.randint(0, len(upblock) - 1)):
                    change_num = random.randint(0, len(upblock) - 1)
                    if new_indi.units[upblock[change_num]].block_type == 3:
                            f = random.random()

                            if f < 0.3:
                                new_indi.units[upblock[change_num]].block_type = 2

        return new_indi

    def change(self, gen_no):
        print("changing")
        new_indi = []
        indi_acc = []
        in_acc = []
        change = 0
        print(self.pops.pop_size)
        print(len(self.pops.individuals))
        for i in range(self.pops.pop_size):
            if self.pops.individuals[i].cv == 1:
                indi_acc.append(self.pops.individuals[i].acc)
                indi_acc.sort()
            else:
                in_acc.append(self.pops.individuals[i].acc)
                in_acc.sort()

        for i in range(self.pops.pop_size):
            indi = self.pops.individuals[i]

            if len(indi_acc) != 0:
                threshold = indi_acc[-1]

            else:
                threshold = in_acc[int(0.4*len(in_acc))]

            if indi.acc >= threshold and indi.cv != 1:
                new=[]
                error = []
                for i in range(30):
                    ind = self.change_indi(indi)
                    new.append(ind)
                    net = cnn.EvoCNNModel(new[i])
                    param = utils.StatusUpdateTool.get_total_params(net)
                    error.append(np.abs(param -self.pops.min_complexity))
                print(sorted(error), np.abs(indi.complexity-self.pops.min_complexity))
                list_error = np.argsort(error)

                if (error[list_error[0]] < 1/10 * self.pops.min_complexity) or (gen_no < 4 and error[list_error[0]]< np.abs(indi.complexity-self.pops.min_complexity)):
                    new[list_error[0]].change = 1
                    new_indi.append(new[list_error[0]])
                    change += 1
                    print("change:%s" %(new[list_error[0]].id))

                else:
                    new_indi.append(indi)
            else:
                new_indi.append(indi)
        self.pops.individuals = new_indi
        print(change)
        return change




