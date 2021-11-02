import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

window.onload = async () => {
  const xs = [1, 2, 3, 4];
  const ys = [1, 3, 5, 7];

  // 数据可视化
  // scatterplot 散点图
  tfvis.render.scatterplot(
    { name: '线性回归样本' },
    { values: xs.map((x, i) => ({ x, y: ys[i] })) },
    { xAxisDomain: [0, 5], yAxisDomain: [0, 10] } // 设置x,y轴长度
  );

  // 创造连续模型 - 这一层的输入一定是上一层的输出
  const model = tf.sequential();
  // dense 全链接层;units 神经元个数
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  // 设置损失函数和优化器 0.1是学习率
  model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) });

  // 将训练数据转为Tensor
  const inputs = tf.tensor(xs);
  const labels = tf.tensor(ys);
  // 调用训练方法
  await model.fit(inputs, labels, {
    batchSize: 1, // 设置小批量数据量
    epochs: 100, // 迭代整个训练数据的次数
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss']),
  });
};
