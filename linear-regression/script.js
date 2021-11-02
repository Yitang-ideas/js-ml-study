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
    batchSize: 4, // 设置小批量数据量
    epochs: 200, // 迭代次数
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss']),
  });

  // 预测x为5的时候y值多少
  const output = model.predict(tf.tensor([5]));
  // output.print();
  // console.log(output.dataSync());
  alert(`x 为 5 时，预测 y 值为 ${output.dataSync()[0]}`);
};
