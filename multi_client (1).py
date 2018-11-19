from fl_client import FederatedClient
import datasource1
import datasource2
import datasource3
import datasource4
import datasource5
import multiprocessing
import threading

def start_client1():
    print("start client1")
    c = FederatedClient("172.17.0.2", 1111, datasource1.Mnist)
def start_client2():
    print("start client2")
    c = FederatedClient("172.17.0.2", 1111, datasource2.Mnist)
def start_client3():
    print("start client3")
    c = FederatedClient("172.17.0.2", 1111, datasource3.Mnist)
def start_client4():
    print("start client4")
    c = FederatedClient("172.17.0.2", 1111, datasource4.Mnist)
def start_client5():
    print("start client5")
    c = FederatedClient("172.17.0.2", 1111, datasource5.Mnist)
    
    
if __name__ == '__main__':
    jobs = []
    tar = "start_client"
    for i in range(1,6):
        # threading.Thread(target=start_client).start()
        if i==1:
            p = multiprocessing.Process(target=start_client1)
        if i==2:
            p = multiprocessing.Process(target=start_client2)
        if i==3:
            p = multiprocessing.Process(target=start_client3)
        if i==4:
            p = multiprocessing.Process(target=start_client4)
        if i==5:
            p = multiprocessing.Process(target=start_client5)       
        jobs.append(p)
        p.start()
    # TODO: randomly kill