import numpy as np
from matplotlib.colors import ListedColormap


class AdalineGD(object):
    # eta: float  # 学習率　0-1
    # n_iter: int  # training data 訓練データの訓練回数
    # random_state: int  # 重みを初期化するための乱数シード

    # classに値追加
    def __init__(self, eta=0.01, n_iter=500, random_state=100):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(
            self.random_state)  # 乱数の再現性を担保　randomstateは乱数調整表のように、ランダムではあるが何回実行しても同じ値を示す。seedと同意
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # 正規分布を使った乱数調整　重みベクトルを作成
        # X.shape = [100 ,2] 100-訓練データの数, 2-特徴量の数（2要素）
        # loc Avr, scale 標準偏差　size 出力配列のサイズ
        self.cost_ = []
        # 誤差平方和のコスト関数
        for i in range(self.n_iter):  # 訓練回数
            net_input = self.net_input(X)  # 予測された値
            output = self.activation(net_input)  # activation関数は今は空っぽなので、output = net_inputでも可
            errors = (y - output)  # 誤差
            # yが実際の値　outputは結果　残差平方和なので、残差(誤差)を求める
            self.w_[1:] += self.eta * X.T.dot(errors)  # X.T.dot 転置して内積を求める
            self.w_[0] += self.eta * errors.sum()  # エラーの合計　0だと0
            cost = (errors ** 2).sum()  # 残差平方和
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        # n.dot ベクトルの内積
        # 実際の要素Xとランダムに作成された数値の内積を取り、それにランダムで作成した値をたす。それが0以上であれば1...
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


import matplotlib.pyplot as plt
import pandas as pd
import os

s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

# 1行2列に変換
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# ADALINEを実装する前に、データを標準化してみる
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# ADALINE実装　訓練回数10 学習率0.01
ada1 = AdalineGD(n_iter=500, eta=0.01)
ada1.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada1)
plt.title("adaline - Gradient Descent")
plt.xlabel('sepal length [std]')
plt.ylabel('petal length [std]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
# plt.show()

# エポック数(試行回数)とコスト（残差平方和）の関係
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=500, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()
