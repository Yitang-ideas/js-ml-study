import * as tf from '@tensorflow/tfjs';

// // 0 维数组
// const t0 = tf.tensor(1);
// t0.print();
// console.log(t0);

// // 1维
// const t1 = tf.tensor([1, 2]);
// t1.print();
// console.log(t1);

// // 2维
// const t2 = tf.tensor([
//   [1, 2, 3],
//   [3, 4, 5],
// ]);
// t2.print();
// console.log(t2);

// // 3维
// const t3 = tf.tensor([[[1]]]);
// t3.print();
// console.log(t3);

const input = [1, 2, 3, 4];
// 存储每个神经元权重，且与输入层相对应
const w = [
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6],
  [4, 5, 6, 7],
];
// 输出层初始化为0
const output = [0, 0, 0, 0];

// 传统for循环
for (let i = 0; i < w.length; i++) {
  for (let j = 0; j < input.length; j++) {
    output[i] += input[j] * w[i][j];
  }
}

console.log(output);

// tensorflow 处理
tf.tensor(w).dot(tf.tensor(input)).print();
