import numpy as np
import re


class ABtest:
    def __init__(self,string):
        self.string = string

    def cuped(self):
        output = (self.string).replace(' ', '')
        #print(f"""{ABtest.remove_spaces.__name__}() ==> {output}""")
        #self.string = output
        return ABtest(output)

    def remove_special_chars(self):
        output = re.sub("[^A-Za-z0-9]", "", self.string)
        #print(f"""{ABtest.remove_special_chars.__name__}() ==> {output}""")
        #self.string = output
        return ABtest(output)

    def buckets(self):
        output = self.string.lower()
        #print(f"""{ABtest.lowercase.__name__}() ==> {output}""")
        #self.string = output
        return ABtest(output)

    def bootstrap(self):
    # Creates a pipeline with a list of functions


        return {
            "effect_control_group_size": 1,
            "effect_target_group_size": 1,
            f"effect_significance": np.random.randint(0,2)
        }