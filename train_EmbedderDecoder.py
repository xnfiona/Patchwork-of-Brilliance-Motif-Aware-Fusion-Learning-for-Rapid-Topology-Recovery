#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/28 5:43 下午
# @Site    : 
# @File    : train_EmbedderDecoder.py
# @Software: PyCharm
from processtools import *
from GCNEmbedderDecoder import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run_training_demo():

    file = open('merge_super_graph_list.pkl', 'rb')
    G_list = pickle.load(file)
    G_score_list = pickle.load(file) #
    file.close()

    G_start = G_list[0]
    best_score = max(G_score_list)
    print("supre graph score:{}, best score:{}".format(G_score_list, best_score))
    original_degrees = [G_start.degree[i] for i in range(G_start.number_of_nodes())]
    top_k = 200
    epochs = 20000
    embedder = GCNEmbedder(in_channels=2, hidden_channels=32, out_channels=64)
    decoder = GraphDecoder(h_dim=64, num_nodes=len(G_start.nodes()))
    data_list = []
    adj_true = []
    for G in G_list:
        temp = gcn_graph(G)
        data_list.append(temp)
        adj_true.append(torch.tensor(nx.to_numpy_array(G), dtype=torch.float32))
    adj_true = torch.stack(adj_true)

    # # === Training ===
    # optimizer = torch.optim.Adam(list(embedder.parameters()) + list(decoder.parameters()), lr=0.01)
    # min_loss = float('Inf')
    # for epoch in range(epochs):
    #     embedder.train()
    #     decoder.train()
    #     h_batch = torch.stack([embedder(data) for data in data_list])
    #     pred_adj = decoder(h_batch)
    #     loss = F.binary_cross_entropy(pred_adj, adj_true)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if (epoch + 1) % 20 == 0:
    #         print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    #     if min_loss > loss:
    #         min_loss = loss
    #         torch.save(embedder.state_dict(), "merge_embedder_{}.pth".format(epochs))
    #         torch.save(decoder.state_dict(), "merger_decoder_{}.pth".format(epochs))
    #         print("已保存更低loss:{}, epoch:{}".format(loss.item(), epoch+1))
    #

    # === Evaluation + Visualization ===
    embedder.load_state_dict(torch.load("merge_embedder_{}.pth".format(epochs)))
    decoder.load_state_dict(torch.load("merger_decoder_{}.pth".format(epochs)))
    embedder.eval()
    decoder.eval()
    with torch.no_grad():
        h_batch = torch.stack([embedder(data) for data in data_list])
        adj_batch = decoder(h_batch)
    G_pred_score_list = []
    for i in range(len(G_list)):
        G_true = G_list[i]
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        nx.draw(G_true, pos=nx.kamada_kawai_layout(G_true), with_labels=True, node_color='lightgreen')
        plt.title("Original Graph")

        plt.subplot(1, 2, 2)
        # G_pred = build_topk_graph(adj_batch[i], top_k, original_degrees, G_start)  # 保证连通,度数差不多, 需要保证节点的位置和邻居信息
        pred_bin = (adj_batch[i].numpy() > 0.5).astype(int)
        G_pred = nx.from_numpy_array(pred_bin)

        nx.draw(G_pred, pos=nx.kamada_kawai_layout(G_pred), with_labels=True, node_color='lightblue')
        plt.title("Predicted Graph")
        plt.show()
        # 比较边集合的差异
        edge_true = set(G_true.edges())
        edge_pred = set(G_pred.edges())
        common_edges = edge_true & edge_pred
        print("相同的边的数量：{}".format(len(common_edges)))
        print("相同边的集合：{}".format(common_edges))
        # 比较score
        score_true = G_score_list[i]
        score_pred = calRM(G_pred, G_start)
        G_pred_score_list.append(score_pred)
        print("当前预测节点{}的true score和 predict score分别是{},{},{}, {}".format(i, score_true, score_pred, nx.is_connected(G_pred), len(G_pred.edges())))
    print("finished!!!")


def run_training_demo_more():

    file1 = open('merge_super_graph_list.pkl', 'rb')
    G_list1 = pickle.load(file1)
    G_score_list1 = pickle.load(file1) #
    file1.close()

    file2 = open('merge_super_graph_list_new_200node.pkl', 'rb')
    G_list2 = pickle.load(file2)
    G_score_list2 = pickle.load(file2)  #
    file2.close()

    G_start1 = G_list1[0]
    best_score1 = max(G_score_list1)
    print("supre graph1 score:{}, best score:{}".format(G_score_list1, best_score1))
    original_degrees1 = [G_start1.degree[i] for i in range(G_start1.number_of_nodes())]

    G_start2 = G_list2[0]
    best_score2 = max(G_score_list2)
    print("supre graph2 score:{}, best score:{}".format(G_score_list2, best_score2))
    original_degrees2 = [G_start2.degree[i] for i in range(G_start2.number_of_nodes())]

    top_k1 = 100
    top_k2 = 200
    epochs = 20000
    max_n = 200
    embedder = GCNEmbedder(in_channels=2, hidden_channels=32, out_channels=64)
    decoder = GraphDecoder(h_dim=64, num_nodes=max_n)
    data_list = []
    adj_true = []
    for G in G_list1:
        G = pad_networkx_graph(G, max_n)
        temp = gcn_graph(G)
        data_list.append(temp)
        adj_true.append(torch.tensor(nx.to_numpy_array(G), dtype=torch.float32))
    for G in G_list2:
        G = pad_networkx_graph(G, max_n)
        temp = gcn_graph(G)
        data_list.append(temp)
        adj_true.append(torch.tensor(nx.to_numpy_array(G), dtype=torch.float32))
    adj_true = torch.stack(adj_true)
    #
    # # === Training ===
    # optimizer = torch.optim.Adam(list(embedder.parameters()) + list(decoder.parameters()), lr=0.01)
    # min_loss = float('Inf')
    # for epoch in range(epochs):
    #     embedder.train()
    #     decoder.train()
    #     h_batch = torch.stack([embedder(data) for data in data_list])
    #     pred_adj = decoder(h_batch)
    #     loss = F.binary_cross_entropy(pred_adj, adj_true)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if (epoch + 1) % 20 == 0:
    #         print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    #     if min_loss > loss:
    #         min_loss = loss
    #         torch.save(embedder.state_dict(), "merge_embedder_more{}.pth".format(epochs))
    #         torch.save(decoder.state_dict(), "merger_decoder_more{}.pth".format(epochs))
    #         print("已保存更低loss:{}, epoch:{}".format(loss.item(), epoch+1))


    # === Evaluation + Visualization ===
    embedder.load_state_dict(torch.load("merge_embedder_more{}.pth".format(epochs)))
    decoder.load_state_dict(torch.load("merger_decoder_more{}.pth".format(epochs)))
    embedder.eval()
    decoder.eval()
    with torch.no_grad():
        h_batch = torch.stack([embedder(data) for data in data_list])
        adj_batch = decoder(h_batch)
    G_pred_score_list = []
    num1 = len(G_list1)
    num2 = len(G_list2)
    for i in range(num1):
        G_true = G_list1[i]
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        nx.draw(G_true, pos=nx.kamada_kawai_layout(G_true), with_labels=True, node_color='lightgreen')
        plt.title("Original Graph")

        plt.subplot(1, 2, 2)
        # G_pred = build_topk_graph(adj_batch[i], top_k, original_degrees, G_start)  # 保证连通,度数差不多, 需要保证节点的位置和邻居信息
        pred_bin = (adj_batch[i][:top_k1, :top_k1].numpy() > 0.5).astype(int)
        G_pred = nx.from_numpy_array(pred_bin)

        nx.draw(G_pred, pos=nx.kamada_kawai_layout(G_pred), with_labels=True, node_color='lightblue')
        plt.title("Predicted Graph")
        plt.show()
        # 比较边集合的差异
        edge_true = set(G_true.edges())
        edge_pred = set(G_pred.edges())
        common_edges = edge_true & edge_pred
        print("相同的边的数量：{}".format(len(common_edges)))
        print("相同边的集合：{}".format(common_edges))
        # 比较score
        score_true = G_score_list1[i]
        score_pred = calRM(G_pred, G_start1)
        G_pred_score_list.append(score_pred)
        print("当前预测节点{}的true score和 predict score分别是{},{},{}, {}".format(i, score_true, score_pred, nx.is_connected(G_pred), len(G_pred.edges())))

    for i in range(num2):
        G_true = G_list2[i]
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        nx.draw(G_true, pos=nx.kamada_kawai_layout(G_true), with_labels=True, node_color='lightgreen')
        plt.title("Original Graph")

        plt.subplot(1, 2, 2)
        # G_pred = build_topk_graph(adj_batch[i], top_k, original_degrees, G_start)  # 保证连通,度数差不多, 需要保证节点的位置和邻居信息
        pred_bin = (adj_batch[i+num1][:top_k2, :top_k2].numpy() > 0.5).astype(int)
        G_pred = nx.from_numpy_array(pred_bin)

        nx.draw(G_pred, pos=nx.kamada_kawai_layout(G_pred), with_labels=True, node_color='lightblue')
        plt.title("Predicted Graph")
        plt.show()
        # 比较边集合的差异
        edge_true = set(G_true.edges())
        edge_pred = set(G_pred.edges())
        common_edges = edge_true & edge_pred
        print("相同的边的数量：{}".format(len(common_edges)))
        print("相同边的集合：{}".format(common_edges))
        # 比较score
        score_true = G_score_list2[i]
        score_pred = calRM(G_pred, G_start2)
        G_pred_score_list.append(score_pred)
        print("当前预测节点{}的true score和 predict score分别是{},{},{}, {}".format(i, score_true, score_pred,
                                                                         nx.is_connected(G_pred), len(G_pred.edges())))

    print("finished!!!")


if __name__ == "__main__":
    # run_training_demo()
    run_training_demo_more()
