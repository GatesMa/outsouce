import csv
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import copy


class vrp():
    def __init__(self, num, dis, dem, carLoad, edge_max, loop_max, speed, uTime):
        """
        数据读入，参数是文件名和读入类型，

        """

        # 数据读入，参数是文件名和读入类型，
        self.num = int(num)  #
        self.dis = dis  #
        self.dem = dem  #
        self.carLoad = carLoad  #
        self.edge_max = edge_max  #
        self.loop_max = loop_max  #
        self.speed = speed  #
        self.uTime = uTime  #

        self.edge_max = float(edge_max)
        self.loop_max = float(loop_max)
        self.speed = float(speed)
        self.uTime = float(uTime)

        for i in self.dis:
            for j in range(len(i)):
                i[j] = float(i[j])  # 字符串转成浮点
        for i in self.dem:
            i = float(i)
        carLoad2 = {}
        for i in self.carLoad.keys():
            carLoad2[float(i)] = carLoad[i]
        self.carLoad = carLoad2
        self.tem = [self.num, self.dis, self.dem, self.carLoad, self.edge_max, self.loop_max, self.speed, self.uTime]
        # 临时变量，方便查看读取情况
        print('tem:', self.tem)
        print('================')
        print('num:', self.num)
        print('dis:', self.dis)
        print('dem:', self.dem)
        print('carLoad:', self.carLoad)
        print('edge_max:', self.edge_max)
        print('loop_max', self.loop_max)
        print('speed:', self.speed)
        print('uTime:', self.uTime)
        print('================')



    def init_unit(self, full=True):
        # full控制是否随机填入点，设置为false可以生成空单位
        """
        随机初始化一个个体
        """
        c2r = {}
        c2r[0] = [set(), 0, 0, 0]  # 已在解中的点集，全局满载率，总路程，最长完成时间，
        for i in self.carLoad.keys():
            c2r[i] = {}
            c2r[i]['num'] = self.carLoad[i]  # 记录i型车的实际数量
            c2r[i]['arranged'] = set()  # 记录i型车辆全部路径里包含的所有配送点
            for j in range(c2r[i]['num']):
                c2r[i][j] = [[0, 0], 0, 0, 0]  # 路径，载重量，路程，完成时间

        if (full):
            vers = (np.random.permutation(self.num) + 1)[0: 1 + int(np.random.randint(self.num) / 2)]

            for v in vers:
                tres = self.vertex_insert(c2r, v)  # 将v加入c2v集合并更新除c2v[0]之外的应变动的值，
                # self.info_update(c2r) #更新c2r[0]中的信息
                # 这里直接把update函数放到insert和del函数里面

        return c2r

    def info_update(self, unit):
        """
        根据各种类型车辆路径信息，对unit[0]中的信息进行更新
        同时删除各车型中为空的路径
        """
        car_types = list(unit.keys() - [0])
        tmpSet = set()
        curLoad = 0
        curCap = 0
        path_dis = 0
        maxTime = 0
        for i in car_types:
            # 点集取并集
            tmpSet = tmpSet | unit[i]['arranged']

            # 删除该车型下的空路线，空路线后的路线依次前移
            tunit = unit[i].copy()
            del tunit['num']
            del tunit['arranged']

            # 取非空路线
            tvals = [j for j in tunit.values() if j[1] > 0]

            for idx, rout in enumerate(tvals):
                unit[i][idx] = rout

            for j in range(len(tvals), len(tunit)):
                del unit[i][j]

            del tunit

            route_id = list(unit[i].keys() - ['num', 'arranged'])
            route_len = len(route_id)

            # 这里不能直接乘起来加，可能有空路线，空路线最后迭代完成时再处理
            # curCap += i * len(route_id)
            # 其实在做过上面的删除空路线操作后已经可以乘起来加了，但懒得改了

            # 累计更新实际载重，路径长度，最大载重量
            for j in route_id:
                # 这里用载重量非空来判断，这就要求后面修改路径的时候必须修改载重量。实际上unit[0]之外的信息
                # 都必须在增或删点之后一起修改
                if unit[i][j][1] != 0:
                    curLoad += unit[i][j][1]
                    path_dis += unit[i][j][2]
                    curCap += i

            # 最长时间和上面3个属性不一样，需要考虑到最终虚拟车辆叠加路径的问题

            for j in range(unit[i]['num']):
                tmpTime = 0
                # 这里直接用等差数列来累加

                # 这里不用考虑为空的问题，arange可以处理,ind会变成空
                ind = np.arange(j, route_len, unit[i]['num'])
                for k in ind:
                    tmpTime += unit[i][k][3]
                if tmpTime > maxTime:
                    maxTime = tmpTime

        unit[0] = [tmpSet, round(curLoad / curCap, 5) if curCap != 0 else 0, round(path_dis, 5), round(maxTime, 5)]

        return unit

    def vertex_insert(self, unit, ver):
        """
        把ver这个配送点随机分配进给定个体unit的一条路径中
        如果给定个体的车辆路线全部不可分配，则随机选择一种载重量大于self.dem[ver]的车型，加入虚拟车辆

        加入之后要更新具体车辆路线的信息，但不用更新unit[0]中的信息

        返回代码：
        0：unit中已存在ver
        1：ver需求量大于最大装载量
        2：成功加入已有路线
        3：无法单独创立路线并加入（ver与0点距离过长，不满足单边限制或回路限制）
        4：单独创立路线并已加入
        """
        if ver in unit[0][0]:
            return 0  # 已经存在，插入失败

        # 如果不存在，那么要么1、随机选择一条路插入 2、随机选一种车型，新建路径并插入

        # 单点需求量大于最大载量，报错
        if self.dem[ver] > max(list(self.carLoad.keys())):
            # raise Exception("{}点的货物需求量超过当前最大货车载量{}!\n")
            return 1

        # 路径不好用异常处理，只能在随机到之后再看

        car_list = np.array(list(self.carLoad.keys()))
        car_list = car_list[car_list > self.dem[ver]]  # 可以承载ver点需求量的车列表

        route_aff = []
        for i in car_list:
            for j in list(unit[i].keys() - ['num', 'arranged']):
                # 这里没法根据路径长来排除路径，因为不知道用ver和哪个点的距离来比
                # 剩余载量大于ver点的需求载量
                if i - unit[i][j][1] >= self.dem[ver]:
                    route_aff.append((i, j))
        # 保证随机性，将疑似可加入的路径顺序打乱
        random.shuffle(route_aff)

        for i, j in route_aff:
            troute = unit[i][j].copy()
            tpath = troute[0]
            tload = troute[1]
            tdis = troute[2]
            ttime = troute[3]

            tinds = np.random.permutation(len(tpath) - 1)  # 可插入位置随机排序

            for k in tinds:
                pre_node = tpath[k]
                post_node = tpath[k + 1]
                # 对i型车，j号路径，k位置后尝试插入
                # 这里只进行单边检测和路径总长度检测，因为载重量在之前加入route_aff的时候就检测过了
                if self.dis[pre_node][ver] < self.edge_max and self.dis[post_node][ver] < self.edge_max \
                        and tdis - self.dis[pre_node][post_node] + self.dis[pre_node][ver] + self.dis[post_node][
                    ver] < self.loop_max:
                    # 满足要求，在k和k+1之间加入ver点
                    tpath.insert(k + 1, ver)
                    tload += self.dem[ver]
                    tdis += self.dis[pre_node][ver] + self.dis[post_node][ver] - self.dis[pre_node][post_node]
                    ttime = (len(tpath) - 2) * self.uTime + tdis / self.speed

                    unit[i][j] = [tpath, round(tload, 5), round(tdis, 5), round(ttime, 5)]
                    unit[i]['arranged'].add(ver)

                    self.info_update(unit)

                    return 2

        # 执行到这里都没有return，说明有满足载量>dem[ver]的车型，但是这些车型里的任何已有路径都满了
        # 新加入路径

        # 检测是否可以新加
        if self.dis[ver][0] > self.edge_max or self.dis[ver][0] * 2 > self.loop_max:
            return 3

        # 可以新加
        tar_car = car_list[np.random.randint(0, len(car_list))]  # 随机选车型

        tind = len(unit[tar_car].keys()) - 2  # 不算num和arranged的2个长度
        unit[tar_car][tind] = [[0, ver, 0], self.dem[ver], round(self.dis[0][ver] * 2, 5), \
                               round(self.uTime + (self.dis[0][ver] * 2) / self.speed, 5)]
        unit[tar_car]['arranged'].add(ver)

        self.info_update(unit)
        return 4

    def vertex_del(self, unit):
        """
        随机删除unit个体中的一个点
        删除之后更新除unit[0]之外的信息
        """
        if len(unit[0][0]) == 0:
            raise Exception("试图从空路径中删除点！")

        car_list = np.array(list(self.carLoad.keys()))

        route_aff = []

        # 实际上可以确定，因为在插入和删除操作后都有update操作
        for i in car_list:
            for j in list(unit[i].keys() - ['num', 'arranged']):
                # 这里我没法确定有没有空路径，所以只能遍历加入
                if unit[i][j][2] > 0:  # 这里很奇怪，有时候会有e-15级别的数进来
                    route_aff.append((i, j))
        # 随机取一条路
        ti, tj = route_aff[np.random.randint(len(route_aff))]

        troute = unit[ti][tj]
        tpath = troute[0]
        tload = troute[1]
        tdis = troute[2]
        ttime = troute[3]

        # 随机选点
        tind = np.random.randint(1, len(tpath) - 1)
        pre_node = tpath[tind - 1]
        post_node = tpath[tind + 1]
        tar_node = tpath[tind]

        tpath.remove(tar_node)

        tload = round(tload - self.dem[tar_node], 10)
        tdis = round(tdis - self.dis[tar_node][pre_node] - \
                     self.dis[tar_node][post_node] + self.dis[pre_node][post_node], 10)

        ttime = round(tdis / self.speed + (len(tpath) - 2) * self.uTime, 10)

        unit[ti][tj] = [tpath, round(tload, 5), round(tdis, 5), round(ttime, 5)]
        unit[ti]['arranged'].remove(tar_node)

        self.info_update(unit)

        # 返回删除的点的序号，方便删除后再加入
        return tar_node

    def init_population(self, num):
        """
        初始化给定尺寸的种群

        给定数量num，初始化出合法的包含num个个体的种群
        输出: list  num x 1
        """
        population = []
        for i in range(num):
            population.append([self.init_unit()])

        return population

    def get_score(self, unit, fac_full, fac_time, fac_distance):
        """
        获得给定个体的适应度

        给定一个个体unit，输出这个个体对应的适应度（越低说明越好）
        fac_full, fac_time, fac_distance分别为满载率系数，运送时长系数，运送距离系数，系数越大说明该
        项越重要。
        """

        # 系数归一化，保留比例即可
        factors = np.array([fac_full, fac_distance, fac_time])
        factors = factors / factors.sum()

        full_load = unit[0][1]

        empty_per = 1 - full_load  # 满载率越高越好，路程长度和时间越小越好，这里把满载率变为空载率
        # 方便后面的计算
        path_dis = unit[0][2]
        finish_time = unit[0][3]

        if (path_dis == 0):
            return np.inf
        # 总系数，代表最重要的因素：尽可能地多加点进去。
        ver_fac = pow(2, (self.num - len(unit[0][0]) + 1))

        # val = ver_fac * (factors * np.log(np.array([empty_per, path_dis, finish_time]) + 100)).sum()
        # val = ver_fac * (factors * (np.array([empty_per * 10, path_dis, finish_time]))).sum() 不行
        val = ver_fac * (factors * np.log(np.array([empty_per + 0.1, path_dis, finish_time]))).sum()
        return val

    def mutate(self, unit):
        """
        产生给定个体的变异后代

        给定一个个体unit，输出一个unit的合法变异个体
        """

        # 这里必须用深拷贝
        unit_m = copy.deepcopy(unit)
        if (len(unit_m[0][0]) < 2):
            return unit_m

        # 这里暂且使用这种简单的不均匀分布来做，不均匀的目的是尽可能少修改原来的染色体

        ###续写标记， 这里tlen为1的时候下面choice函数会报错
        tlen = len(unit_m[0][0])
        chg_num = int(np.random.choice(np.arange(1, tlen), 1, \
                                       p=np.arange(1, tlen)[::-1] / np.arange(1, tlen).sum()))

        for i in range(chg_num):
            self.vertex_del(unit_m)

        # 到底这里是把删掉的重新加回去，还是取相同数量的没加的点再加回去，根据之后的结果再看
        # 不行，感觉结果不好，不能固定加入原来删除的点，这样解太单一了
        need_vers = np.random.permutation(list(set(np.arange(1, self.num + 1)) - unit_m[0][0]))
        for i in range(chg_num):
            resu = self.vertex_insert(unit_m, need_vers[i])
            # 这里无法插入其实也可以理解，如果有的点只能和其他连起来凑到路径里，
            # 那它取出来之后就是不能单独再放进去

            # if resu != 2 and resu != 4:
            #     raise Exception("取出的点无法插入！")

        # 以上操作保持一个路径内点的总数不变，但初始化不一定能把所有点都加入，因此下面要随机加入点

        # 不行，看起来不能随机加，要加得加和0点距离大于edge_max的
        # if len(unit_m[0][0]) < self.num:
        #     tall = set(np.arange(1, self.num + 1))
        #     tnow = unit_m[0][0]
        #     tver = list(tall - tnow)[np.random.randint(len(tall - tnow))]
        #     self.vertex_insert(unit_m, tver)

        hard_vers = set(np.arange(0, self.num + 1)[np.array(self.dis[0]) > self.edge_max])
        # 有难搞的点没加进去
        # 这里难搞的点其实还可以按照难搞的程度进行排序，但是先不弄
        if len(hard_vers - unit_m[0][0]) > 0:
            tar_ver = np.random.choice(list(hard_vers - unit_m[0][0]))
            self.vertex_insert(unit_m, tar_ver)

        self.info_update(unit_m)

        return unit_m

    def solve(self, start_num=50, max_iter=50, max_population=1000, fac_full=5,
              fac_distance=4, fac_time=3, show=True, dia=True):
        """

        利用遗传算法搜寻最优解，迭代结束后输出最优个体（目标值最小的个体）
        """
        # 遗传参数
        # start_num = 50
        # iter_num = 50
        # max_p = 1000

        # #控制种群进化方向的参数，这里写出比例关系即可，在之后会做系数归一化
        # fac_full = 5
        # fac_distance = 4
        # fac_time = 3

        # 初始化种群
        cur_p = np.array(self.init_population(start_num)).reshape(start_num)
        num_p = start_num

        # 下面2个变量可以后期画图使用
        min_val = []
        nums = []
        num_vers = []

        for i in range(max_iter):

            data = np.c_[[self.get_score(i, fac_full, fac_time, fac_distance) for i in cur_p], cur_p]
            data = data[data[:, 0].argsort()]  # 按得分升序排，分越低越好
            if show:
                print("迭代轮数：{}  最佳满载率：{:.2f}  最短路程：{:.2f}  最短时间：{:.2f}  已分配点数：{}  得分：{:.2f}  种群数量：{}" \
                      .format(i, data[0][1][0][1], data[0][1][0][2], data[0][1][0][3], \
                              len(data[0][1][0][0]), data[0][0], data.shape[0]))
            # 控制种群数量，防止cpu爆炸
            if data.shape[0] > max_population:
                data = data[0: max_population]

            num = data.shape[0]

            min_val.append(data[0][0])
            nums.append(num)
            num_vers.append(len(data[0][1][0][0]))

            # 选择存活
            prob = np.random.rand(num)
            prob_ = list(map(lambda x: (pow(num, x / num) - 1) / (num - 1), \
                             np.arange(1, num + 1)))  # 概率阈值，最大适应度的阈值是1
            ind = prob > prob_
            data = data[ind, :]

            # 选择变异
            num = data.shape[0]
            prob = np.random.rand(num)
            prob_ = np.arange(1, num + 1) / num
            ind = prob > prob_
            for j in data[ind]:
                data = np.r_[data, [[0, self.mutate(j[1])]]]

            cur_p = data[:, 1]

        # 最后看能不能安排路线到不同车
        final_ans = self.init_unit(full=False)
        cars = np.array(sorted(self.carLoad.keys()))
        tep = cur_p[0]
        for i in (tep.keys() - [0]):
            for j in (tep[i].keys() - ['num', 'arranged']):
                des_car = cars[np.where((cars >= tep[i][j][1]) == True)[0][0]]
                final_ans[des_car][len(final_ans[des_car]) - 2] = tep[i][j]
                for v in tep[i][j][0]:
                    if v != 0:
                        final_ans[des_car]['arranged'].add(v)

        self.info_update(final_ans)
        min_val.append(self.get_score(final_ans, fac_full, fac_time, fac_distance))
        print("调整路线后结果： 最佳满载率：{:.2f}  最短路程：{:.2f}  最短时间：{:.2f}  已分配点数：{}  得分：{:.2f}  " \
              .format(final_ans[0][1], final_ans[0][2], final_ans[0][3], \
                      len(final_ans[0][0]), self.get_score(final_ans, fac_full, fac_time, fac_distance)))
        # self.ans_analysis(final_ans)

        if (dia):
            plt.subplots_adjust(hspace=1)

            plt.subplot(311)
            plt.plot(nums)
            plt.title("Size of population")
            plt.xlabel("iterations")

            plt.subplot(312)
            plt.plot(min_val)
            plt.title("Best value")
            plt.xlabel("iterations")

            plt.subplot(313)
            plt.plot(num_vers)
            plt.title("Number of vertexs in best route")
            plt.xlabel("iterations")

            plt.show()

        return self.ans_analysis(final_ans)

    def ans_analysis(self, unit):
        """
        详细分析给定的unit个体，用于solve完成搜寻之后输出最终解的各种信息
        相当于main_reg中的get_ans函数

        输出：
        车辆（序号及载重量）及其对应路线 car2route
        每次路线满载率list
        总体满载率
        总配送时间
        各点配送完毕时间
        各线路完成时间（车辆回到配送中心时间）
        总路程
        """
        print("车辆型号及对应路线信息：\n")
        for i in (unit.keys() - [0]):
            print("\n载重{}的车辆共有{}辆，其需行走的路线及相应运载量，路程长度，耗时如下:\n".format(i, self.carLoad[i]))
            for j in (unit[i].keys() - ['num', 'arranged']):
                print("   路线：", unit[i][j][0],
                      "  运载量：{:.2f}  路线长度：{:.2f} km  耗时：{:.2f} min".format(unit[i][j][1], unit[i][j][2],
                                                                           unit[i][j][3]))
        print("\n全局满载率：{:.2f}\n全局路程长度：{:.2f} km\n全局运送耗时：{:.2f} min\n".format(unit[0][1], unit[0][2], unit[0][3]))

        p2t = {}
        for i in (unit.keys() - [0]):
            if len(unit[i]) - 2 <= unit[i]['num']:  # 如果路径数小于车辆数，那每条路径都可以分到一辆车
                for j in (unit[i].keys() - ['num', 'arranged']):
                    tpath = unit[i][j][0]
                    tlen = len(tpath)
                    ttime = 0
                    for k in range(1, tlen):
                        ttime += self.uTime + self.dis[tpath[k - 1]][tpath[k]] / self.speed
                        p2t[tpath[k]] = ttime
            else:
                # 路径数大于车辆数，那就得叠加了
                tlen = len(unit[i]) - 2
                for j in range(unit[i]['num']):
                    tinds = np.arange(j, tlen, unit[i]['num'])
                    unit[i][j] = [unit[i][k] for k in tinds]
                    ttime = 0
                    for troute in unit[i][j]:
                        for k in range(1, len(troute[0]) - 1):
                            ttime += self.uTime + self.dis[troute[0][k - 1]][troute[0][k]] / self.speed
                            p2t[troute[0][k]] = ttime
                        ttime += self.dis[troute[0][-1]][troute[0][-2]] / self.speed
                for j in range(len(unit[i]) - 3, unit[i]['num'] - 1, -1):
                    del unit[i][j]

        # 这里暂时没有考虑虚拟路线点分配的情况，只是简单考虑没有虚拟路线的情况
        print("各配送点配送完毕时间：\n")
        for i in range(1, self.num + 1):
            if i in p2t:
                t = str(round(p2t[i], 2))
            else:
                t = "未送达！"
            print("{}号点：".format(i) + t)

        unit['overall'] = unit.pop(0)
        print("-----------------------------------------------------")
        print(unit)
        print("-----------------------------------------------------")
        with open('res.json', 'w') as fi:
            json.dump(unit, fi, cls=MyEncoder)
        return unit

def cal(x, y):  # 计算
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5

def tran(coo):  # 计算
    dis = []
    llen = len(coo)
    for i in range(llen):
        dis.append([])
        for j in range(llen):
            dis[i].append(cal(coo[i], coo[j]))
    return dis

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == '__main__':
    # 读入问题数据
    v = vrp('data/data_coo.csv', 'coo')

    # start_num是初始化种群数量，max_iter是迭代次数，max_population是允许的种群最大数量
    # fac_full, fac_time, fac_distance分别为满载率系数，运送时长系数，运送距离系数，系数越大说明该项越重要
    # show控制是否实时打印数据,dia控制是否显示可视化过程
    ans = v.solve(start_num=150, max_iter=100, max_population=1500, fac_full=5,
                  fac_distance=4, fac_time=3, show=True, dia=True)
    print("############")
    ans = json.dumps(ans, cls=MyEncoder)
    # print(type(ans))
    # print(ans)
