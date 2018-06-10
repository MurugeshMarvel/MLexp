import pandas
import math

class decision_tree():
    def __init__(self, dataset, target_col="last"):
        self.dataset = dataset
        self.target_col = target_col
    def read_file(self, header_names):
        self.data = pandas.read_csv(self.dataset,names=header_names)
        self.data_len = len(self.data)
        return self.data

    def target_classes(self):
        col_len = len(data_frame.columns.values)
        if self.target_col == "last":
            target_col_name = data_frame.columns.values[col_len-1]
            self.target_class_val = []
            self.target_class_count = {}
            for i in self.data[target_col_name]:
                if i not in self.target_class_val:
                    self.target_class_val.append(i)
                    self.target_class_count[i] = 1
                else:
                    self.target_class_count[i] += 1
            print (self.target_class_count)
        print (self.target_class_val)
        self.input_attribute = list(data_frame.columns.values)
        del self.input_attribute[col_len-1]
        #print (self.input_attribute)

    def entropy(self):
        def summation(list):
            sum = 0
            for i in range(len(list)):
                sum += list[i]
            return sum
        entropy_list =[]
        for i in self.target_class_count.keys():
            proportionality = float(self.target_class_count[i]/self.data_len)
            temp_ = -1.0 * proportionality * float(math.log(proportionality,2))
            entropy_list.append(temp_)
        print (summation(entropy_list))

dec_tree = decision_tree(dataset = "data/sample.data")
#data_frame = dec_tree.read_file(["buying", "maint", "doors", "persons", "lug_boot", "safety"])
data_frame = dec_tree.read_file(["outlook", "temperature", "humidity", "wind", "playtennis"])
dec_tree.target_classes()
dec_tree.entropy()
