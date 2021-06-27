import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

#为GAT核心模块
'''
3个比较核心的参数：
seq:指的是输入的节点特征矩阵，大小为 [num_graph, num_node, fea_size]
out_sz: 指的是变换后的节点特征维度，也就是Wh-后的节点表示维度
bias_mat 是经过变换后的邻接矩阵，大小为 [num_node, num_node]


作者首先将原始节点特征 seq 进行变换得到了 seq_fts。这里，作者使用卷积核大小为 1 的 1D 卷积模拟投影变换，
投影变换后的维度为 out_sz。注意，这里投影矩阵W是所有节点共享，所以 1D 卷积中的多个卷积核也是共享的。
seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
也就是说，seq_fts 的大小为 [num_graph, num_node, out_sz]。

回顾前面的公式展开aT[hi||hj]=a1Thi+a2Thj也可以认为是投影变换，只不过投影到 1 维表示。注意，这里节点及其邻居的投影是分开的，有两套投影参数a1,a2 ，对应下面两个 conv1d 中的参数。
f_1 = tf.layers.conv1d(seq_fts, 1, 1)
f_2 = tf.layers.conv1d(seq_fts, 1, 1)


经过 tf.layers.conv1d(seq_fts, 1, 1) 之后的 f_1 和 f_2 维度均为 [num_graph, num_node, 1]。

将 f_2 转置之后与 f_1 叠加，通过广播得到的大小为 [num_graph, num_node, num_node] 的 logits，就是一个注意力矩阵：
logitsij=(a->T[whi->||whj->])

按照 GAT 的公式，我们只要对 logits 进行 softmax 归一化就可以拿到注意力权重 ，也就是代码里的 coefs。但是，这里为什么会多一项 bias_mat 呢？
coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
因为的 logits 存储了任意两个节点之间的注意力值，但是，归一化只需要对每个节点的所有邻居的注意力进行
所以，引入了 bias_mat 就是将 softmax 的归一化对象约束在每个节点的邻居上，如下式的红色部分。

bias_mat 是如何实现的呢
'''
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

