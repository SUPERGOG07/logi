import json
from math import log

from graphviz import Digraph


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    return datasets, labels


# 计算当前节点的数据集的熵
def calc_entropy(datasets):
    data_size = len(datasets)
    label_counts = {}
    for data in datasets:
        label = data[-1]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    entropy = 0.0
    for count in label_counts.values():
        prob = count / data_size
        entropy -= prob * log(prob, 2)
    return entropy


# 计算当前节点的某个特征的一个条件熵
def calc_conditional_entropy(datasets, axis, value):
    subset = [data[:axis] + data[axis + 1:] for data in datasets if data[axis] == value]
    return calc_entropy(subset), subset


# 计算当前节点的某个特征的信息增益
def calc_information_gain(datasets, axis):
    base_entropy = calc_entropy(datasets)
    values = set([data[axis] for data in datasets])
    conditional_entropy = 0.0
    for value in values:
        conditional_entropy += calc_conditional_entropy(datasets, axis, value)[0]
    information_gain = base_entropy - conditional_entropy
    return information_gain


# 选择当前节点最适合特征
def choose_best_feature(datasets):
    feature_count = len(datasets[0]) - 1
    best_feature_index = -1
    best_information_gain = 0.0
    for i in range(feature_count):
        information_gain = calc_information_gain(datasets, i)
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_feature_index = i
    return best_feature_index


# 生成ID3决策树
def create_decision_tree(datasets, labels):
    class_list = [data[-1] for data in datasets]
    # 如果节点数据分类一致，则使其成为叶节点
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 如果没有更多的特征可供划分，选择出现次数最多的类别作为最终的分类结果
    if len(datasets[0]) == 1:
        label_counts = {}
        for data in datasets:
            label = data[-1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        sorted_label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_label_counts[0][0]

    best_feature_index = choose_best_feature(datasets)
    best_feature_label = labels[best_feature_index]
    decision_tree = {best_feature_label: {}}
    del (labels[best_feature_index])
    feature_values = set([data[best_feature_index] for data in datasets])
    for value in feature_values:
        sub_labels = labels[:]
        decision_tree[best_feature_label][value] = create_decision_tree(
            calc_conditional_entropy(datasets, best_feature_index, value)[1], sub_labels)
    return decision_tree


def predict(decision_tree, labels, data):
    root = list(decision_tree.keys())[0]
    sub_tree = decision_tree[root]
    label_index = labels.index(root)
    for key in sub_tree.keys():
        if data[label_index] == key:
            if isinstance(sub_tree[key], dict):
                class_label = predict(sub_tree[key], labels, data)
            else:
                class_label = sub_tree[key]
    return class_label


# 获取所有节点中最多子节点的叶节点
def getMaxLeafs(myTree):
    numLeaf = len(myTree.keys())
    for key, value in myTree.items():
        if isinstance(value, dict):
            sum_numLeaf = getMaxLeafs(value)
            if sum_numLeaf > numLeaf:
                numLeaf = sum_numLeaf
    return numLeaf


def plot_model(tree, name):
    g = Digraph("G", filename=name, format='png', strict=False)
    g.attr('node', fontname='SimSun')
    g.attr('edge', fontname='SimSun')
    first_label = list(tree.keys())[0]
    g.node("0", first_label)
    _sub_plot(g, tree, "0")
    leafs = str(getMaxLeafs(tree) // 10)
    g.attr(rankdir='LR', ranksep=leafs)
    g.view()


root = "0"
def _sub_plot(g, tree, inc):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            g.node(root, list(tree[first_label][i].keys())[0])
            g.edge(inc, root, str(i))
            _sub_plot(g, tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            g.node(root, tree[first_label][i])
            g.edge(inc, root, str(i))


if __name__ == '__main__':
    datasets, labels = create_data()
    decision_tree = create_decision_tree(datasets, labels.copy())
    print("决策树：")
    print(json.dumps(decision_tree, indent=4, ensure_ascii=False))

    # 绘制树状图
    plot_model(decision_tree, "tree")

    data = ['青年', '否', '否', '一般']
    print(data)
    result = predict(decision_tree, labels.copy(), data)
    print("预测结果：", result)
