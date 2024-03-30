import networkx as nx
import numpy as np
import random
from collections import deque
import time
num_episodes = 500
learning_rate = 0.03
gamma = 0.5
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01

class SFCMappingEnvironment:
    def __init__(self):
        # Thiết lập mạng vật lý
        self.PHY_nodes = list(PHY.nodes())
        self.PHY_weights_node = [PHY.nodes[node]['weight'] for node in self.PHY_nodes]
        self.PHY_array = nx.adjacency_matrix(PHY).toarray()

        # Thiết lập mạng SFC
        self.SFC_nodes = []
        self.SFC_weights_node = []
        self.SFC_array = []

        # Thiết lập không gian trạng thái và hành động
        self.state_space = []
        self.action_space = list(range(len(self.PHY_nodes)))
         # Phương thức cập nhật trọng số của các node PHY sau khi mapping
    def update_PHY_node_weights(self, mapping_pairs):
        for node_SFC, node_PHY in mapping_pairs:
            self.PHY_weights_node[node_PHY] -= self.SFC_weights_node[node_SFC]

def create_random_sfc():
    # Số lượng nút từ 3 đến 6
    #random.seed(100)
    num_nodes = random.randint(3, 6)
    
    # Trọng số nút từ 5 đến 30
    node_weights = [random.randint(5, 30) for _ in range(num_nodes)]
    
    # Tạo đồ thị
    sfc = nx.Graph()
    
    # Thêm các nút vào đồ thị
    for i in range(num_nodes):
        sfc.add_node(i, weight=node_weights[i])
    
    # Trọng số cạnh từ 5 đến 60
    edge_weights = [random.randint(5, 60) for _ in range(num_nodes - 1)]
    
    # Thêm các cạnh vào đồ thị
    for i in range(num_nodes - 1):
        sfc.add_edge(i, i + 1, weight=edge_weights[i])
    
    return sfc

def find_max_values(array):
    # Đếm số hàng và số cột của mảng
    num_rows = len(array)
    num_cols = len(array[0])
    
    max_values = []  # Danh sách để lưu trữ giá trị lớn nhất của mỗi hàng
    max_positions = []  # Danh sách để lưu trữ vị trí của giá trị lớn nhất của mỗi hàng

    chosen_columns = set()  # Tập hợp các cột đã được chọn
    
    for row_idx in range(num_rows):
        max_value = None  # Giá trị lớn nhất của hàng hiện tại
        max_position = None  # Vị trí của giá trị lớn nhất trong hàng hiện tại
        
        for col_idx in range(num_cols):
            if col_idx not in chosen_columns:
                value = array[row_idx][col_idx]  # Giá trị của phần tử hiện tại
                
                if max_value is None or value > max_value:
                    max_value = value
                    max_position = (row_idx, col_idx)

        if max_value is not None:
            chosen_columns.add(max_position[1])  # Thêm cột đã chọn vào tập hợp
            
            max_values.append(max_value)  # Thêm giá trị lớn nhất vào danh sách
            max_positions.append(max_position)  # Thêm vị trí vào danh sách

    return max_values, max_positions

def dijkstra(graph, start, end, weight_requirement):
    num_nodes = len(graph)
    distances = np.full(num_nodes, np.inf)  # Khởi tạo khoảng cách ban đầu là vô cùng
    distances[start] = 0  # Khoảng cách từ nút bắt đầu đến chính nó là 0

    visited = set()  # Tập các nút đã được duyệt
    previous = np.full(num_nodes, None)  # Mảng lưu các nút trước đó trên đường đi ngắn nhất
    updated_graph = [list(row) for row in graph]
    # Duyệt qua tất cả các nút
    for _ in range(num_nodes):
        # Tìm nút có khoảng cách nhỏ nhất và chưa được duyệt
        min_distance = np.inf
        min_node = None
        for node in range(num_nodes):
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node

        if min_node is None:
            break  # Không có đường đi từ nút bắt đầu đến nút kết thúc

        visited.add(min_node)  # Đánh dấu nút đã được duyệt

        # Kiểm tra nếu đã đến nút kết thúc
        if min_node == end:
            path = []
            node = end
            while node is not None:
                path.insert(0, node)
                node = previous[node]
                
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                updated_graph[node1][node2] -= weight_requirement

            total_weight = 0
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                total_weight += graph[node1][node2]
                
            return len(path), updated_graph, total_weight#, updated_graph  # Trả về số lượng nút phải đi qua, đường đi và tổng trọng số cạnh

        # Cập nhật khoảng cách và nút trước đó cho các nút kề
        for neighbor in range(num_nodes):
            if neighbor not in visited and graph[min_node][neighbor] >= weight_requirement:
                new_distance = distances[min_node] + graph[min_node][neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = min_node

    return -1, updated_graph, 0#, graph  # Không tìm thấy đường đi từ nút bắt đầu đến nút kết thúc

#================================================================================================================
#=========================================== Deep Q learning ====================================================
#================================================================================================================

class DQNAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.num_episodes = 2000
        self.learning_rate = 0.03
        self.gamma = 0.5
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.update_target_nn = 10
        self.batch_size = 32
        self.hidden_dim = 32
        self.replay_buffer_size = 2000
        
        # Tạo mạng main network và target network
        self.main_weights1, self.main_bias1, self.main_weights2, self.main_bias2 = self.build_network(len(self.state_space), self.hidden_dim, len(self.action_space))
        self.target_weights1, self.target_bias1, self.target_weights2, self.target_bias2 = self.build_network(len(self.state_space), self.hidden_dim, len(self.action_space))
        
        # Sao chép trọng số của mạng main vào mạng target
        self.update_target_network()

        # Bộ nhớ tái trải nghiệm (replay buffer)
        self.replay_buffer = deque(maxlen=5000)

    def build_network(self, input_dim, hidden_dim, output_dim):
        np.random.seed(1)
        
        weights1 = np.random.randn(hidden_dim, input_dim) # 32 x state_space
        bias1 = np.zeros((hidden_dim, 1)) # 32 x 1
        weights2 = np.random.randn(hidden_dim, output_dim) # 32 x action_space
        bias2 = np.zeros((output_dim, 1)) # action_space x 1 
        
        return weights1, bias1, weights2, bias2 
    def relu(self, x):
        return np.maximum(0, x)

    def linear(self, x):
        return x

    # Predict cho trạng thái X
    def forward_propagation(self, X, is_main=True):
        # Predict bằng mạng main
        if is_main:
            hidden_layer_input = np.dot(self.main_weights1, X.T) + self.main_bias1 # 32 x 20
            hidden_layer_output = self.relu(hidden_layer_input) #32 x 20
            output_layer_input = np.dot(self.main_weights2.T , hidden_layer_output) + self.main_bias2 # 15 x 20
        # Predict bằng mạng target
        else:
            hidden_layer_input = np.dot(self.target_weights1, X.T ) + self.target_bias1 # 32 x 20
            hidden_layer_output = self.relu(hidden_layer_input) # 32 x 20
            output_layer_input = np.dot(self.target_weights2.T , hidden_layer_output) + self.target_bias2 #15 x 20
            
        output_layer_output = self.linear(output_layer_input)
        
        return output_layer_output.T # 20 x 15
    
    # Đạo hàm và cập nhật trọng số mạng
    def backward_propagation(self, X, y_true, y_pred):
        m = X.shape[0] 
        d_output_layer = (2 / m) * (y_pred - y_true)# 20 x 15
        #print("d_output_layer: ", d_output_layer.shape)
        d_weights2 = 1/m * np.dot(self.relu(np.dot(self.main_weights1, X.T) + self.main_bias1), d_output_layer) #32 x 15
        d_bias2 = 1/m * np.sum(d_output_layer, axis=0, keepdims=True).T #15 x 1
        
        d_hidden_layer = np.dot(self.main_weights2, d_output_layer.T) * (self.relu(np.dot(self.main_weights1, X.T) + self.main_bias1) > 0) # 
        d_weights1 = 1/ m * np.dot(d_hidden_layer, X) #(32 x 20) x (20 x 3)=(32, 3)
        d_bias1 = 1/ m * np.sum(d_hidden_layer.T, axis=0, keepdims=True).T # (32 x 1)

        # Cập nhật trọng số cho mạng main
        self.main_weights1 -= self.learning_rate * d_weights1 # 32 x 3
        self.main_bias1 -= self.learning_rate * d_bias1 # 32x1
        self.main_weights2 -= self.learning_rate * d_weights2 # 32 x 15
        self.main_bias2 -= self.learning_rate * d_bias2 # 15 x 1
        
    def update_target_network(self):
        self.target_weights1 = self.main_weights1.copy()
        self.target_bias1 = self.main_bias1.copy()
        self.target_weights2 = self.main_weights2.copy()
        self.target_bias2 = self.main_bias2.copy()
        
    def remember_exp(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def learn_from_replay(self, ):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Predict cho trạng thái hiện tại và trạng thái kế
        q_values_current = self.forward_propagation(states, is_main=True)
        q_values_next = self.forward_propagation(next_states, is_main=False)

        targets = q_values_current.copy()
        batch_indices = np.arange(self.batch_size)

        targets[batch_indices, actions] = rewards + self.gamma * np.max(q_values_next, axis=1) * (1 - dones)

        # Huấn luyện mạng chính
        self.backward_propagation(states, targets, q_values_current)
        
    
    def choice_action(self, current_state_array, current_action_space):  
        # Khám phá
        if np.random.rand() < agent.epsilon:
            # Chọn hành động bất kì trong không gian hành động 
            action = np.random.choice(current_action_space) 
            current_action_space.remove(action) 
        # Khai thác
        else:
            q_values = self.forward_propagation(current_state_array, is_main=True)
            action = np.argmax(q_values[0])
        return action, current_action_space
    
    def deep_q_learning(self,env2):
        # Quá trình training
        for episode in range(self.num_episodes):
            current_state_space2 = env2.state_space.copy()# Copy không gian trạng thái
            current_action_space2 = env2.action_space.copy() # Copy không gian hành động
            current_PHY_array2 = env2.PHY_array.copy()# Copy ma trận kề của PHY

            selected_action_space2 = np.array([]) # Lưu trữ các hành động đã chọn
            selected_state_array2 = np.array([]) # Lưu trữ các state đã chọn
            
            done2 = False
            while not done2:
                if len(current_state_space2) == 0: # Check xem đã hết trạng thái chưa
                    break 
                # Chọn trạng thái
                current_state2 = current_state_space2.pop(0) # Lấy trạng thái từ trong mảng
                current_state_array2 = np.eye(len(env2.state_space))[current_state2]
                selected_state_array2 = np.append(selected_state_array2, current_state2)# lưu trạng thái vừa chọn vào mảng để tính toán reward
        
                # Chọn hành động (chọn node PHY) thỏa mãn cap of PHY node > cap of SFC node và link map giữa 2 node PHY > link giữa 2 SFC liên tiếp
                while 1:
                    if len(current_action_space2) == 0:
                        print("can't map ")
                        return current_PHY_array2, 0
                    # Chọn hành động theo khám phá hoặc khai thác
                    action2, current_action_space2 = agent.choice_action(current_state_array2, current_action_space2)
                    # Kiểm tra hành động được chọn ở trên (action)
                    if action2 not in selected_action_space2 and env2.PHY_weights_node[action2] >= env2.SFC_weights_node[current_state2]: # Kiểm tra hành động đó có được chọn trước đo hay chưa và cap của SFc >= cap của PHY
                        if len(selected_state_array2) == 1: # Nếu là hành động động đầu tiên được chọn thì được chọn và lưu trữ
                            selected_action_space2 = np.append(selected_action_space2,action2)
                            break
                        else: # Nếu là hành động thứ 3 trở đi được chọn
                            num_hop2, current_PHY_array2, total_weight_edge = dijkstra(current_PHY_array2, int(selected_action_space2[-1])  , action2, env2.SFC_array[current_state2 - 1][current_state2]) # Tính toán xem có đường đi có thỏa mãn trọng số và cập nhật ma trận kề của PHY khi đã tìm thấy đường đi ngắn nhất
                            if num_hop2 != - 1: # Nếu có đường đi thì hành động sẽ được chọn và lưu trữ
                                selected_action_space2 = np.append(selected_action_space2,action2)
                                break
                        
                if len(selected_state_array2) == 1: # Tính toán Reward cho hành động đầu tiên
                    reward2 = 200 - (env2.PHY_weights_node[action2] - env2.SFC_weights_node[current_state2])
                elif 1 < len(selected_state_array2) <  len(env2.state_space) :  # Không phải trạng thái đầu và trạng thái cuối
                    reward2 = 200 - (env2.PHY_weights_node[action2] - env2.SFC_weights_node[current_state2]) - 5 * num_hop2 
                    
                else: # Tính toán Reward cho hành động từ lần thứ 2
                    done2 = True
                    reward2 = 200 - (env2.PHY_weights_node[action2] - env2.SFC_weights_node[current_state2]) - 5 * num_hop2 
                next_state2 = current_state2 + 1 if current_state_space2 else current_state2
                next_state_array2 = np.eye(len(env2.state_space))[next_state2]
                # Lưu trữ vào buffer
                self.remember_exp(current_state_array2, action2, reward2, next_state_array2, done2)
            # Học từ bộ nhớ tái trải nghiệm   
            self.learn_from_replay()
             # Giảm epsilon sau mỗi episode
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            # Cập nhật mạng target network sau một số episodes
            if episode % self.update_target_nn == 0 and episode != 0:
                self.update_target_network()
        
        return current_PHY_array2, 1 

#================================================================================================================
#============================================== Q learning ======================================================
#================================================================================================================

def calculator_q(current_state_space, q_table, selected_action_space, current_state, action, reward):
    if current_state_space: # Tính Q_value cho trạng thái không phải cuối cùng
        # Copy mảng trạng thái kế
        array_next_state = q_table[current_state_space[0]].copy()
        # Tìm giá trị max của trạng thái tiếp theo mà không có các hàng động đã chọn
        max_next_state = np.max(array_next_state[np.logical_not(np.isin(np.arange(len(array_next_state)), selected_action_space))])
        # Tính Q_value
        q_table[current_state][action] = (1 - learning_rate) * q_table[current_state][action] + learning_rate * (reward + gamma * max_next_state)
    else: # Tính Q_value cho trạng thái cuối cùng
        q_table[current_state][action] = (1 - learning_rate) * q_table[current_state][action] + learning_rate * reward
        
# Thuật toán Q-Learning
def q_learning(env, num_episodes, gamma, epsilon,epsilon_min, epsilon_decay):
    # Khởi tạo bảng Q table là ma trận 3(số node SFC) x 6(số node PHY) có tất cả giá trị bằng 0
    q_table = np.zeros((len(env.state_space), len(env.action_space)))
    # Quá trình training
    for episode in range(num_episodes):
        current_state_space = env.state_space.copy()# Copy không gian trạng thái
        current_action_space = env.action_space.copy() # Copy không gian hành động
        current_PHY_array = env.PHY_array.copy()# Copy ma trận kề của PHY

        selected_action_space = np.array([]) # Lưu trữ các hành động đã chọn
        selected_state_array = np.array([]) # Lưu trữ các state đã chọn
        while 1:
            if len(current_state_space) == 0: # Check xem đã hết trạng thái chưa
                break 
            # Chọn trạng thái
            current_state = current_state_space.pop(0) # Lấy trạng thái từ trong mảng
            selected_state_array = np.append(selected_state_array, current_state)# lưu trạng thái vừa chọn vào mảng để tính toán reward
            # Chọn hành động (chọn node PHY) thỏa mãn cap of PHY node > cap of SFC node và link map giữa 2 node PHY > link giữa 2 SFC liên tiếp
            while 1:
                # Khám phá
                if len(current_action_space) == 0:
                    print("can't map ")
                    return q_table, current_PHY_array, 0
                if np.random.rand() < epsilon:
                    action = np.random.choice(current_action_space) # Chọn hành động bất kì trong không gian hành động 
                    current_action_space.remove(action) 
                # Khai thác
                else:
                    action = np.argmax(q_table[current_state]) # Chọn hành động có giá trị Q lớn nhất trong bảng Q table
                # Kiểm tra hành động được chọn ở trên (action)
                if action not in selected_action_space and env.PHY_weights_node[action] >= env.SFC_weights_node[current_state]: # Kiểm tra hành động đó có được chọn trước đo hay chưa và cap của SFc >= cap của PHY
                    if len(selected_state_array) == 1: # Nếu là hành động động đầu tiên được chọn thì được chọn và lưu trữ
                        selected_action_space = np.append(selected_action_space,action)
                        break
                    else: # Nếu là hành động thứ 3 trở đi được chọn
                        
                        num_hop, current_PHY_array, total_weight_edge = dijkstra(current_PHY_array, int(selected_action_space[-1])  , action, env.SFC_array[current_state - 1][current_state]) # Tính toán xem có đường đi có thỏa mãn trọng số và cập nhật ma trận kề của PHY khi đã tìm thấy đường đi ngắn nhất
                        
                        if num_hop != - 1: # Nếu có đường đi thì hành động sẽ được chọn và lưu trữ
                            selected_action_space = np.append(selected_action_space,action)
                            break
                
            if len(selected_state_array) == 1: # Tính toán Reward cho hành động đầu tiên
                reward = 200 - (env.PHY_weights_node[action] - env.SFC_weights_node[current_state])
                calculator_q(current_state_space, q_table, selected_action_space, current_state, action, reward)
            else: # Tính toán Reward cho hành động từ lần thứ 2
                reward = 200 - (env.PHY_weights_node[action] - env.SFC_weights_node[current_state]) - 5 * num_hop
                calculator_q(current_state_space, q_table, selected_action_space, current_state, action, reward)  
            
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
    return q_table, current_PHY_array, 1 # Trả về bảng Q_table

#================================================================================================================
#================================================= RUNNING ======================================================
#================================================================================================================

PHY = nx.read_gml("D:/CODE LEARN CODE BLOCK/PYTHON/REINFORCEMENT LEARNING AND NETWORKING/network/Code/data_import/data_PHY/atlanta.gml")#./data_import/data_PHY/atlanta.gml

interact = 10
num_SFC_embed = 20

time_run_q_learning_arr = []
time_run_deep_q_learning_arr = []
AR_q_learning = []
AR_deep_q_learning = []

for i in range(interact):
    count1  = 1
    count2  = 0
    count3  = 0
    time_run_q_learning = 0
    time_run_deep_q_learning = 0
    env1 = SFCMappingEnvironment()
    env2 = SFCMappingEnvironment()
    for j in range(num_SFC_embed):
        #print(f"============================================ SFC thu: {count1} =================================")
        SFC = create_random_sfc()
        # SFC cho thuật toán Q learning
        env1.SFC_nodes = list(SFC.nodes())
        env1.SFC_weights_node = [SFC.nodes[node]['weight'] for node in env1.SFC_nodes]
        env1.SFC_array = nx.adjacency_matrix(SFC).toarray()
        env1.state_space = list(range(len(env1.SFC_nodes)))
        
        # SFC cho thuật toán Q learning
        env2.SFC_nodes = list(SFC.nodes())
        env2.SFC_weights_node = [SFC.nodes[node]['weight'] for node in env2.SFC_nodes]
        env2.SFC_array = nx.adjacency_matrix(SFC).toarray()
        env2.state_space = list(range(len(env2.SFC_nodes)))
           
        start_time1 = time.time()
        #print("env.SFC_weights_node: ", env2.SFC_weights_node)
        #print("env.SFC_array: ", env2.SFC_array)
        # Gọi vào thuật toán Q learning
        q_table, current_PHY_array1, check1 = q_learning(env1, num_episodes, gamma, epsilon,epsilon_min, epsilon_decay)
        if check1 == 0:
            print(f"======================================= SFC {count2} can't mapping by Q learning=================================")
        else:
            count2 = count2 + 1 
            print(f"======================================= SFC {count2} is mapping by Q learning  =================================")
            max_values, mapping_pairs = find_max_values(q_table)
            
            #print("max_positions: ", mapping_pairs)  
            env1.update_PHY_node_weights(mapping_pairs)
            env1.PHY_array = current_PHY_array1

        end_time1 = time.time()
        
        time_run_q_learning += end_time1 - start_time1

        #Gọi vào thuật toán deep Q learning
        start_time2 = time.time()
        agent = DQNAgent(env2.state_space, env2.action_space)
        current_PHY_array2, check2 = agent.deep_q_learning(env2)
        
        if check2 == 0:
            print(f"======================================= SFC {count3} can't mapping by Deep Q learning=================================")
            
        # Nếu SFC được mapping bằng Deep Q learning
        else:
            count3 = count3 + 1
            print(f"======================================= SFC {count3} is mapping by Deep Q learning  =================================")
            states = np.eye(len(env2.state_space))
            q_action = agent.forward_propagation(states, is_main=True)
            max_values, mapping_pairs = find_max_values(q_action)
            #print("max_positions: ", mapping_pairs) 
            
            env2.update_PHY_node_weights(mapping_pairs)
            env2.PHY_array = current_PHY_array2

        end_time2 = time.time()
        time_run_deep_q_learning += end_time2 - start_time2
        count1 = count1 + 1
    acceptation_rate_q_learning = float(count2 / (count1 - 1))
    print("acceptation_rate_q_learning: ", acceptation_rate_q_learning)
    AR_q_learning = np.append(AR_q_learning, acceptation_rate_q_learning)
    acceptation_rate_deep_q_learning = float(count3 / (count1 - 1))
    print("acceptation_rate_deep_q_learning: ", acceptation_rate_deep_q_learning)
    AR_deep_q_learning = np.append(AR_deep_q_learning, acceptation_rate_deep_q_learning)
    
    print("time_run_q_learning: ", time_run_q_learning)
    time_run_q_learning_arr = np.append(time_run_q_learning_arr, time_run_q_learning)
    print("time_run_deep_q_learning: ", time_run_deep_q_learning)
    time_run_deep_q_learning_arr = np.append(time_run_deep_q_learning_arr, time_run_deep_q_learning)
        
np.savetxt('D:/CODE LEARN CODE BLOCK/PYTHON/REINFORCEMENT LEARNING AND NETWORKING/network/Code/Q-deep-q/data-atlanta/60/AR_deep_q_learning.txt', AR_deep_q_learning) # Xuất giá trị ra file
np.savetxt('D:/CODE LEARN CODE BLOCK/PYTHON/REINFORCEMENT LEARNING AND NETWORKING/network/Code/Q-deep-q/data-atlanta/60/AR_q_learning.txt', AR_q_learning) # Xuất giá trị ra file
np.savetxt('D:/CODE LEARN CODE BLOCK/PYTHON/REINFORCEMENT LEARNING AND NETWORKING/network/Code/Q-deep-q/data-atlanta/60/time_run_deep_q_learning.txt', time_run_deep_q_learning_arr) # Xuất giá trị ra file
np.savetxt('D:/CODE LEARN CODE BLOCK/PYTHON/REINFORCEMENT LEARNING AND NETWORKING/network/Code/Q-deep-q/data-atlanta/60/time_run_q_learning.txt', time_run_q_learning_arr) # Xuất giá trị ra file


#np.savetxt('./Q-deep-q/data-pioro40/60/AR_deep_q_learning.txt', AR_deep_q_learning) # Xuất giá trị ra file
#np.savetxt('./Q-deep-q/data-pioro40/60/AR_q_learning.txt', AR_q_learning) # Xuất giá trị ra file
#np.savetxt('./Q-deep-q/data-pioro40/60/time_run_deep_q_learning.txt', time_run_deep_q_learning_arr) # Xuất giá trị ra file
#np.savetxt('./Q-deep-q/data-pioro40/60/time_run_q_learning.txt', time_run_q_learning_arr) # Xuất giá trị ra file
