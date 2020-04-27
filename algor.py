import csv
import numpy as np
import json

class vrp():
    def __init__(self, num, dis, dem, carLoad, edge_max, loop_max, speed, uTime):
        #数据读入，参数是文件名和读入类型，
        self.num = num #
        self.dis = dis #
        self.dem = dem #
        self.carLoad = carLoad #
        self.edge_max = edge_max #
        self.loop_max = loop_max #
        self.speed = speed #
        self.uTime = uTime #

        self.edge_max = float(edge_max)
        self.loop_max = float(loop_max)
        self.speed = float(speed)
        self.uTime = float(uTime)

        for i in self.dis:
            for j in range(len(i)):
                i[j] = float(i[j])  # 字符串转成浮点
        
    def get_ans(self):
        """
        此函数用于获取solve()函数的计算结果并转换成问题的答案形式
        
        输出：
        车辆（序号及载重量）及其对应路线 car2route
        每次路线满载率list
        总体满载率
        总配送时间
        各点配送完毕时间
        各线路完成时间（车辆回到配送中心时间）
    
        """
        
        #car2route {车序号:[[路线1]，[路线2]...]}
        #cars保存序号对应载量
        car2route, cars = self.solve()
        
        num_route = 0 #所有路线数
        for i in car2route.values():
            num_route += len(i)
            
        load_capacity = 0 #如果全部满载，能装的货量
        for i in car2route.keys():
            load_capacity += len(car2route[i]) * cars[i]
        
        #全局满载率 = 实际发货量/各回路对应车辆的车载量之和 
        all_load_per = sum(self.dem) / load_capacity
        
        load_per = {} #记录车号对应各趟的满载率
        #形式和car2route差不多
        for i in car2route.keys():
            load_per[i] = []
            for j_route in car2route[i]:
                loaded = 0
                for k in j_route:
                    loaded += self.dem[k]
                load_per[i].append(loaded / cars[i])
                
        #计算各点送达时间和各回路完成时间，回路完成时间最大值就是全局完成时间
        #这里不考虑一辆车回到配送中心后的重新装货时间
        p_time = {} # 点序号：卸载完成时间
        loop_time = {} #车号：回路1完成时间，回路2完成时间
        for i in car2route.keys():
            loop_time[i] = []
            cur_time = 0#每次开始遍历一个车的路程时从头计时，因为车可以并发
            for j_route in car2route[i]:
                for k in range(1, len(j_route) - 1):
                    cur_time += self.dis[j_route[k]][j_route[k - 1]] / self.speed
                    cur_time += self.uTime
                    p_time[j_route[k]] = cur_time
                cur_time += self.dis[0][j_route[len(j_route) - 2]] / self.speed #回到0点
                loop_time[i].append(cur_time)#添加回路完成时间  
        
        # #把car2route的key从车序号改成(车序号，车载量)
        tem_keys = list(car2route.keys())
        for i in tem_keys:
            car2route['({}, {})'.format(i, cars[i])] = car2route[i]
            car2route.pop(i) #json文件转存不支持tuple作字典键，只能放弃
        
        #车辆及对应路线，全局满载率，各回路满载率，全局完成时间，各客户点卸货完成时间，各回路完成时间
        # ans = "车辆及对应路线：{}\n全局满载率：{}\n各回路满载率:{}\n全局完成时间: \
        # {}\n各客户点卸货完成时间:{}\n各回路完成时间:{}\n".format(car2route, all_load_per, \
        #     load_per, max(loop_time.values()), p_time, loop_time)
        # print("车序号对应载货量:\n")
        # for i, j in enumerate(cars):
        #     print("{}:{} ".format(i,j))
        # print("\n(车序号, 载重量): 路线\n")
        # for i in car2route.items():
        #     print(str(i) + '\n')
        
        # print("全局满载率：%f\n" % all_load_per)
        # print("车序号：各路线满载率\n", load_per)
        # print("\n全局完成时间：{}\n".format(max(loop_time.values())))
        # print("点序号：卸货完成时间\n", p_time)
        # print("\n车序号：各回路完成时间\n", loop_time)

        ans = json.dumps({'car2route': car2route, 'overall_load_per': all_load_per, 'load_per': load_per, \
            'finish time': max(loop_time.values()), 'ind of point and corresponding finish time': p_time, \
            'finish time of routes': loop_time})
        print('Get One Request: ', ans)
        return ans
        
        
   
          
    def solve(self):
        '''
        问题求解
        距离统一为km，时间统一为min
        
        输入：对象
        
        输出：各车辆路线安排 car2route
        
        
        '''  

        
        #考虑单边最大因素
        maxx = max(max(self.dis))
        for i in self.dis:
            for j in range(len(i)):
                if i[j] > self.edge_max:
                    i[j] = maxx*10 #保证dis[i][j]的节约值小于0
                    #暂不换算时间，最后路线算出来再统一换算
                    
        #计算节约矩阵
        saves = []
        llen = len(self.dis)
        for i in range(1, llen):
            for j in range(i + 1, llen):
                saves.append((i, j, self.dis[0][i] + self.dis[0][j] - self.dis[i][j]))
        saves.sort(key = lambda x: x[2], reverse = True)#降序排节约值
        #排，但是不删最后的，即使负了很多。因为有可能一个点离其他所有点都很远
        #那属于他的表项都会删掉，但这个点可能离货仓很近。
        #我们只需要保证每条回路和单边长度合格即可。
    
        
        
        
        #根据1月17日答疑问题第8问，此处设计算法优先考虑高满载率，其次考虑时间，最后考虑路程
        
        #满载率 = 所有回路中车辆出发时装货量/所有出动车辆总载量
        #
        
        ###这里先实现一个简单版，就是最原始的cw算法思路，优化目标为路程最短，满载率最高。思路为贪心，可能达不到最优解
        
        car2route = {}
        #(载重量，车序号)：[[0,...,0],[0,,,0]...]
        #车辆及序号对应到list形式的子回路集
        #更正：  车序号：list路线集
        
        p2round = {}#点信息
        #点序号：（车序号，第几趟）
        #各点对应到车辆号以及轮数（一辆车可能跑多轮，轮数用于标记是哪轮）
        
        cars = [] #车辆载重信息表
        for i in self.carLoad.keys():
            for j in range(self.carLoad[i]):
                cars.append(float(i))
        cars.sort(reverse = True)#降序排车，对应优先安排大车的想法
        num_route = [0 for i in range(len(cars))] #记录各车的回路数
        
        for i in range(len(cars)):
            car2route[i] = [] #初始化car2route
        
        ass_num = 0 #已分配的点
        sav_ind = -1 #saves表项游标
        len_sav = len(saves)
        car_ind = -1 #如果当前要加新路线，应该哪辆车承担
        #car_ind变量后续由num_route变量的第一个最小值的位置决定，这样也可以均衡回路数量，并且方便根据回路运载量寻找最合适车辆
        len_car = len(cars)
        
        
        while(ass_num != self.num and sav_ind != len(self.dis)): #这里取最大值遍历len(self.dis)轮可以纳入所有点，或许可以更小。懒得想了
            sav_ind += 1
            #car_ind = (len_car + 1) % len_car
            
            x = saves[sav_ind % len_sav][0] 
            y = saves[sav_ind % len_sav][1] #两个点
            #cost = saves[ind][2]
            
            if x in p2round and y in p2round: #x y均已存在路线中
                if p2round[x] == p2round[y]:
                    continue
                else:
                    continue
                    #这里是说x，y有合并可能，但需要考虑不同车辆的回路合并，有点复杂，暂不实现
            
            elif x in p2round: # x在，y不在
                p_car = p2round[x][0] # x所在回路的车辆序号
                p_round = p2round[x][1] # x所在回路是负责该回路车辆的第几趟
                p_load = cars[p_car] # 运x的车的载重量
                p_route = car2route[p_car][p_round] # x所在回路
                
                
                #如果x位于该趟的与0相邻的位置，且插入后合格，就插入y
                if (p_route[1] == x or p_route[len(p_route) - 2] == x)\
                    and \
                    self.val_check(p_route, x, y, p_load): #插入后不违背单边、总回路、载重量三种限制
                    ass_num += 1 #跳出循环变量
                    if p_route[1] == x: #满足插入检查，x位于1位置
                        car2route[p_car][p_round].insert(1, y)
                        p2round[y] = (p_car, p_round) 
                    else:
                        car2route[p_car][p_round].insert(len(p_route) - 1, y)
                        p2round[y] = (p_car, p_round)
                    
                        
                elif self.dis[0][y] <= self.edge_max and \
                    self.dis[0][y] * 2 <= self.loop_max and self.dem[y] <= cars[0]:
                    # 把y单独算成一个回路，单独开回路
                    
                        tem_ind = 0
                        ass_num += 1
                        for i in cars:#数能容纳[0, y, 0]的车辆有多少辆
                            if i >= self.dem[y]:
                                tem_ind += 1
                                
                        car_ind = num_route.index(min(num_route[0: tem_ind]))#能容纳中已容纳最少的车序号
                        ind_route = len(car2route[car_ind])
                        
                        car2route[car_ind].append([0, y, 0])
                        p2round[x] = (car_ind, ind_route)
                        
                        num_route[car_ind] += 1

                    
                
            elif y in p2round: # y在，x不在
                #情况跟上面一样，只是感觉合并比较麻烦，就重抄一遍
                p_car = p2round[y][0] 
                p_round = p2round[y][1] 
                p_load = cars[p_car]  
                p_route = car2route[p_car][p_round] 
                
                
                
                if (p_route[1] == y or p_route[len(p_route) - 2] == y)\
                    and \
                    self.val_check(p_route, y, x, p_load): #插入后不违背单边、总回路、载重量三种限制
                    
                    ass_num += 1
                    if p_route[1] == y: #满足插入检查，y位于1位置
                        car2route[p_car][p_round].insert(1, x)
                        p2round[x] = (p_car, p_round) 
                    else:
                        car2route[p_car][p_round].insert(len(p_route) - 1, x)
                        p2round[x] = (p_car, p_round)
                    
                #x与y不能合并，尝试把x单独算一条
                elif self.dis[0][x] <= self.edge_max and \
                    self.dis[0][x] * 2 <= self.loop_max and self.dem[x] <= cars[0]:
                    # 把x单独算成一个回路，单独开回路
                    
                        tem_ind = 0
                        ass_num += 1
                        for i in cars:#数能容纳[0, x, 0]的车辆有多少辆
                            if i >= self.dem[x]:
                                tem_ind += 1
                                
                        car_ind = num_route.index(min(num_route[0: tem_ind]))#能容纳中已容纳最少的车序号
                        ind_route = len(car2route[car_ind])
                        
                        car2route[car_ind].append([0, x, 0])
                        p2round[x] = (car_ind, ind_route)
                        
                        num_route[car_ind] += 1


                    
            else: # x y都不在
                
                #如果xy可以一起组成一条回路
                if self.dis_check(x, y) and self.dem[x] + self.dem[y] <= cars[0]:
                    tem_ind = 0
                    for i in cars:
                        if i >= self.dem[x] + self.dem[y]:
                            tem_ind += 1
                    #取能容纳xy货量的所有车中当前路线数最少的车
                    car_ind = num_route.index(min(num_route[0: tem_ind]))
                    ind_route = len(car2route[car_ind])
                    
                    car2route[car_ind].append([0, x, y, 0])
                    p2round[x] = (car_ind, ind_route)
                    p2round[y] = (car_ind, ind_route)

                    ass_num += 2
                    
                    num_route[car_ind] += 1
                    
 
                #x,y即使用最大的车也不能放在一条回路中，则分为两个回路
                else:
                    # 判断x有没有可能成为一条回路
                    if self.dis[0][x] <= self.edge_max and \
                        self.dis[0][x] * 2 <= self.loop_max and self.dem[x] <= cars[0]:
                        
                        tem_ind = 0
                        ass_num += 1
                        for i in cars:#数能容纳[0, x, 0]的车辆有多少辆
                            if i >= self.dem[x]:
                                tem_ind += 1
                                
                        car_ind = num_route.index(min(num_route[0: tem_ind]))#能容纳中已容纳最少的车序号
                        ind_route = len(car2route[car_ind])
                        
                        car2route[car_ind].append([0, x, 0])
                        p2round[x] = (car_ind, ind_route)
                        
                        num_route[car_ind] += 1
                        
                    #判断y有没有可能成为一条回路
                    if self.dis[0][y] <= self.edge_max and \
                        self.dis[0][y] * 2 <= self.loop_max and self.dem[y] <= cars[0]:
                        
                        tem_ind = 0
                        ass_num += 1
                        for i in cars:
                            if i >= self.dem[y]:
                                tem_ind += 1
                        
                        car_ind = num_route.index(min(num_route[0: tem_ind]))
                        ind_route = len(car2route[car_ind])
                        
                        car2route[car_ind].append([0, y, 0])
                        p2round[y] = (car_ind, ind_route)
                        
                        num_route[car_ind] += 1
            
        
        ###
        return car2route, cars
    
    def dis_check(self, x, y):
        
        if self.dis[0][x] <= self.edge_max and self.dis[0][y] <= self.edge_max \
            and self.dis[x][y] <= self.edge_max and \
            self.dis[0][x] + self.dis[0][y] + self.dis[x][y] <= self.loop_max:
            
            return True
        
        return False
    
    
    def val_check(self, route, exist_val, wait_val, max_load):
        # 路径信息list，已存在的点的序号,判断是否可插入的点的序号，路径对应车辆的载重量
        load = 0
        dist = 0
        len_route = len(route)
        
        #载重判断
        for i in route:
            load += self.dem[i] #dem中0虽然出现2次，但dem[0]为0
        if self.dem[wait_val] + load > max_load: #超重
            return False
        
        #回路距离判断  
        for i in range(1, len_route):
            dist += self.dis[route[i]][route[i - 1]] # dis是对称矩阵
        if dist + self.dis[exist_val][wait_val] + self.dis[0][wait_val] > self.loop_max:
            return False
        
        #单边限制
        if self.dis[0][wait_val] > self.edge_max or self.dis[exist_val][wait_val] > self.edge_max:
            return False
        
        return True
        
            
            
        

def cal(x,y):#计算
    return ((x[0] - y[0])**2 + (x[1] - y[1])**2)**0.5
    
def tran(coo):#计算
    dis = []
    llen = len(coo)
    for i in range(llen):
        dis.append([])
        for j in range(llen):
            dis[i].append(cal(coo[i],coo[j]))
    return dis
   

if __name__ == '__main__':
    v = vrp('data/data_dis.csv', 'dis')
    print(v.get_ans())