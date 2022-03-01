import pickle
import torch


def getExampleId(outFname_base, num_example):
    return ("%s-%08d" % (outFname_base, num_example)).encode("UTF-8")

def serializeExample(price, graph):
    price = torch.tensor(float(price), dtype=torch.float32)
    one_data_serialized = pickle.dumps( {"graph": graph, "price" :price})
    return one_data_serialized

def deserializeExample(exampleBytes):
    example_dict = pickle.loads(exampleBytes)
    return example_dict["graph"], example_dict["price"]