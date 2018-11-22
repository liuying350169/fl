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
    c = FederatedClient("172.17.17.3", 5554, datasource1.Mnist)
def start_client2():
    print("start client2")
    c = FederatedClient("172.17.17.3", 5554, datasource2.Mnist)
def start_client3():
    print("start client3")
    c = FederatedClient("172.17.17.3", 5554, datasource3.Mnist)
def start_client4():
    print("start client4")
    c = FederatedClient("172.17.17.3", 5554, datasource4.Mnist)
def start_client5():
    print("start client5")
    c = FederatedClient("172.17.17.3", 5554, datasource5.Mnist)
def start_client6():
    print("start client6")
    c = FederatedClient("172.17.17.3", 5554, datasource1.Mnist)
def start_client7():
    print("start client7")
    c = FederatedClient("172.17.17.3", 5554, datasource2.Mnist)
def start_client8():
    print("start client8")
    c = FederatedClient("172.17.17.3", 5554, datasource3.Mnist)
def start_client9():
    print("start client9")
    c = FederatedClient("172.17.17.3", 5554, datasource4.Mnist)
def start_client0():
    print("start client0")
    c = FederatedClient("172.17.17.3", 5554, datasource5.Mnist)
    
    
if __name__ == '__main__':
    jobs = []
    tar = "start_client"
    for i in range(1,7):
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
        if i==6:
            p = multiprocessing.Process(target=start_client6)
        if i==7:
            p = multiprocessing.Process(target=start_client7)
        if i==8:
            p = multiprocessing.Process(target=start_client8)
        if i==9:
            p = multiprocessing.Process(target=start_client9)
        if i==10:
            p = multiprocessing.Process(target=start_client0)            
        jobs.append(p)
        p.start()
    # TODO: randomly kill

