from fl_client import FederatedClient
import datasource
import multiprocessing
import threading

def start_client1():
    print("start client1")
    c = FederatedClient("172.17.0.2", 1111, datasource.Mnist)
    
    
if __name__ == '__main__':
    jobs = []
    tar = "start_client"
    #for i in range(1,6):
        # threading.Thread(target=start_client).start()
    p = multiprocessing.Process(target=start_client1)     
    jobs.append(p)
    p.start()
    # TODO: randomly kill
