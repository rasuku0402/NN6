#include "nn.h"
#include <time.h>
#include <math.h>

//float型の配列oをxの値で初期化する関数
void init(int n, float x, float*o){
    int r;
    for (r = 0; r <= n - 1;r++){
        o[r] = x;
    }
}

//float型の配列A,x,oを宣言し、o=A×xを代入する関数（Aはm×n行列）
void mul(int m, int n, const float*x,const float*A, float*o){
    int r;
    int l;
    for (l = 0; l <= m - 1; l++){
        for (r = l*n; r <= n-1+l*n; r++){
            o[l] = o[l] + x[r % n] * A[r];
        }
    }
}

//m×n行列A，m行列ベクトルb，n行列ベクトルxに対してAx+bをm行列ベクトルyに代入する関数
void fc(int m, int n, const float *x, const float *A, const float *b, float *o){
    int r;
    init(m, 0, o);
    mul(m, n, x, A, o);
    for (r = 0; r <= m - 1; r++){
        o[r] = o[r] + b[r];
    }
}

//n行列ベクトルxに対して、relu計算を行う関数
void relu(int n,const float *x,float *y){
    for (int i = 0; i < n; i++){
        if(x[i]>0){
            y[i] = x[i];
        }else{
            y[i] = 0;
        }
    }
}

//n行列ベクトルxに対して，最大値を計算する関数
float Max(int n, const float * x){
    int i;
    float j = x[0];
    for (i = 0; i < n; i++){
        if(j < x[i]) {
            j = x[i];
        }
    }
    return j;
}

//n行列ベクトルxに対して，softmax計算を行った関数
void softmax(int n, const float * x, float * y) {
    float sum = 0;
    float x_Max = Max(n, x);
    for (int i = 0; i < n; i++){
        sum = sum + exp(x[i] - x_Max);
    }
    for(int i = 0; i < n; i++){
        y[i] = exp(x[i] - x_Max) / sum;
    }
}

//6層NNによってm×n行列A，m行列ベクトルb，n行列ベクトルxが、inputーfc1ーrelu1ーfc2ーrelu2ーfc3ーsoftmaxーoutputの順で計算され、0～9のいずれかの値を返す関数
int inference6(const float *A1,const float *A2,const float *A3,const float *b1,const float *b2,const float *b3,const float *x,float *y){
    float *y1 = malloc(sizeof(float) * 50);
    float *y2 = malloc(sizeof(float) * 100);
    fc(50,784,x,A1,b1,y1);   //fc1層
    relu(50, y1, y1);   //relu1層
    fc(100, 50, y1, A2, b2, y2);   //fc2層
    relu(100, y2, y2);   //relu2層
    fc(10, 100, y2, A3, b3, y);   //fc3層
    softmax(10, y, y);   //softmax層

    int sub = 0;   //yが最大値の時の値 = ｛0，1，2，3，4，5，6，7，8，9｝のどれか
    float y_max = y[0];
    int a;
    for (a = 0; a < 10; a++){ //yの最大値を求める
        if(y[a]>y_max){
            y_max = y[a];
            sub = a;//yの最大値をsubに代入
        }
    }
    //メモリを開放する
    free(y1);
    free(y2);
    return sub;
}

//n行列ベクトルyと0～9までの整数tを用いてdEdxを計算する、softmaxでの誤差逆伝搬を行う関数
void softmaxwithloss_bwd(int n,const float *y, unsigned char t,float *dEdx){
    for (int i = 0; i < n; i++){
        if(i==t){
            dEdx[i] = y[i] - 1.0;
        }else{
            dEdx[i] = y[i];
        }
    }
}

//n行列ベクトルxと上流からの勾配dEdyを用いてdEdxを計算する、reluでの誤差逆伝搬を行う関数
void relu_bwd(int n,const float *x,const float *dEdy, float *dEdx){
    for (int i = 0; i < n; i++){
        if(x[i]>0){
            dEdx[i] = dEdy[i];
        }else{
            dEdx[i] = 0;
        }
    }
}

//m×n行列Aとm行列ベクトルb、n行列ベクトルxを更新して勾配を求める、fcでの誤差逆伝搬を行う関数
void fc_bwd(int m,int n,const float *x,const float *dEdy,const float *A, float *dEdA, float *dEdb, float *dEdx){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            dEdA[i * n + j] = dEdy[i] * x[j];
        }
    }
    for (int i = 0; i < m; i++){
        dEdb[i] = dEdy[i];
    }
    for (int i = 0; i < n; i++){
        dEdx[i] = 0;
        for (int j = 0; j < m; j++){
            dEdx[i] += A[n * j + i] * dEdy[j];
        }
    }
}

//6層NNでの誤差逆伝搬でありinference6とは逆に、outputーsoftmaxbwdーfcbwd3ーrelubwd2ーfcbwd2ーrelubwd1ーfcbwd1の順で計算される関数
//x,y,A1,A2,A3,b1,b2,b3は順伝搬中での入力、t,dA1,dA2,dA3,db1,db2,db3は逆伝搬中での入力
void backward6(const float *A1,const float *A2,const float *A3,const float *b1,const float *b2,const float *b3,const float *x,unsigned char t,float *y,float *dEdA1,float *dEdA2,float *dEdA3,float *dEdb1,float *dEdb2,float *dEdb3){
    float *y0 = malloc(sizeof(float) * 784);
    float *y1 = malloc(sizeof(float) * 10);
    float *y2 = malloc(sizeof(float) * 100);
    float *y3 = malloc(sizeof(float) * 50);
    float *relu1_x = malloc(sizeof(float) * 50);   //relu1層
    float *fc2_x = malloc(sizeof(float) * 50);   //fc2層
    float *relu2_x = malloc(sizeof(float) * 100);    //relu2層
    float *fc3_x = malloc(sizeof(float) * 100);    //fc3層
    //順伝搬部分
    fc(50, 784, x, A1, b1, relu1_x);
    relu(50, relu1_x, fc2_x);
    fc(100, 50, fc2_x, A2, b2, relu2_x);
    relu(100, relu2_x, fc3_x);
    fc(10, 100, fc3_x, A3, b3, y);
    softmax(10, y, y);
    //逆伝搬部分
    softmaxwithloss_bwd(10, y, t, y1);
    fc_bwd(10, 100, fc3_x, y1, A3, dEdA3, dEdb3, y2);
    relu_bwd(100, relu2_x, y2, y2);
    fc_bwd(100, 50, fc2_x, y2, A2, dEdA2, dEdb2, y3);
    relu_bwd(50, relu1_x, y3, y3);
    fc_bwd(50, 784, x, y3, A1, dEdA1, dEdb1, y0);
    //メモリ開放
    free(y0);
    free(y1);
    free(y2);
    free(y3);
    free(relu1_x);
    free(fc2_x);
    free(relu2_x);
    free(fc3_x);
}

//配列xのn個の要素をランダムに入れ替える関数
void shuffle(int n,int *x){
    int k = 0;
    for (int i = 0; i < n; i++){
        int j= rand()%n;
        k = x[i];
        x[i] = x[j];
        x[j] = k;
    }
}

//推論結果であるyと、訓練データの答えを用いて交差エントロピーEを返す関数
float cross_entropy_error(const float *y,int t){
    return -log(y[t] + 1e-7);
}

//n次元ベクトルxを用いて、配列の加算を行う関数
void add(int n,const float *x,float *o){
    for (int i = 0; i < n; i++){
        o[i] = o[i] + x[i];
    }
}

//n次元ベクトルxを用いて、o[i]=o[i]×x(i=0,1,2,･･･)の計算を行う関数
void scale(int n,float x,float *o){
    for (int i = 0; i < n; i++){
        o[i] = o[i] * x;
    }
}

//[0,1]の範囲の乱数を返す関数
float uniform(void){
    return (float)rand() / ((float)RAND_MAX + 1.0);
}

//ガウス分布の乱数を返す関数（平均a、標準偏差b)
double rand_gauss(double a,double b){
    double c = sqrt(-2.0 * log(uniform())) * sin(2.0 * 3.142 * uniform());
    return a + b * c;
}

//平均0、標準偏差√(2/n)のガウス分布による乱数で、n個の配列oの各要素を初期化
void rand_init_gauss(int n,float *o) {
    float b = sqrt(sqrt(2.0 / n));
    for (int i = 0; i < n; i++){
        o[i] = rand_gauss(0, b);
    }
}

//もしファイルが開かなければ"FILE cannot open"と表示し、そうでなければデータを保存する関数
void save(const char *filename, int m, int n, const float *A, const float *b){
    FILE *fp;
    if((fp=fopen(filename,"wb"))==NULL){
        printf("FILE cannot open");
    }else{
        fwrite(A, sizeof(float), m * n, fp);
        fwrite(b, sizeof(float), m, fp);
        fclose(fp);
    }
}

//もしファイルが開かなければ"FILE cannot open"と表示し、そうでなければデータを読み込む関数
void load(const char *filename, int m, int n, float *A, float *b){
    FILE *fp;
    if((fp=fopen(filename,"rb"))==NULL)
        printf("FILE cannot open\n");
    else{
        fread(A, sizeof(float), m * n, fp);
        fread(b, sizeof(float), m, fp);
        fclose(fp);
    }
}

int main(int argc, char *argv[]){
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);

    //変数を入力する
    int epoch;
    printf("epoch: ");
    scanf("%d", &epoch);
    int n;
    printf("minipatch: ");
    scanf("%d", &n);
    float eta;
    printf("learning rate: ");
    scanf("%f", &eta);

    //すべての値を初期化
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b1 = malloc(sizeof(float) * 50);
    float *b2 = malloc(sizeof(float) * 100);
    float *b3 = malloc(sizeof(float) * 10);
    float *av_dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *av_dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *av_dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *av_dEdb1 = malloc(sizeof(float) * 50 * 100);
    float *av_dEdb2 = malloc(sizeof(float) * 100);
    float *av_dEdb3 = malloc(sizeof(float) * 10);
    float *dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *dEdb1 = malloc(sizeof(float) * 50);
    float *dEdb2 = malloc(sizeof(float) * 100);
    float *dEdb3 = malloc(sizeof(float) * 10);
    int *index = malloc(sizeof(int) * train_count);
    float *y = malloc(sizeof(float) * 10);
    srand(time(NULL));
    rand_init_gauss(784 * 50, A1);
    rand_init_gauss(50 * 100, A2);
    rand_init_gauss(100 * 10, A3);
    rand_init_gauss(50, b1);
    rand_init_gauss(100, b2);
    rand_init_gauss(10, b3);

    //配列indexの要素を、indexの添え字で初期化する
    for (int i = 0; i < train_count; i++){
        index[i] = i;
    }

    //epochの回数繰り返す
    for (int i = 0; i < epoch; i++){
        //配列indexの各要素をランダムシャッフルする
        shuffle(train_count, index);
        //minipatch学習をN/n回繰り返す（Nはindexの要素数、そしてindexから順にｎ個ずつ要素を持ってくる）
        for (int j = 0; j < train_count / n; j++){
            //0で初期化する
            init(784 * 50, 0, av_dEdA1);
            init(50 * 100, 0, av_dEdA2);
            init(100 * 10, 0, av_dEdA3);
            init(50, 0, av_dEdb1);
            init(100, 0, av_dEdb2);
            init(10, 0, av_dEdb3);
            //一つずつ勾配を求める
            for (int k = 0; k < n; k++){
                backward6(A1, A2, A3, b1, b2, b3, train_x + 784 * index[100 * j + k], train_y[index[100 * j + k]], y, dEdA1, dEdA2, dEdA3, dEdb1, dEdb2, dEdb3);
                //勾配を平均勾配に加える
                add(784 * 50, dEdA1, av_dEdA1);
                add(50 * 100, dEdA2, av_dEdA2);
                add(100 * 10, dEdA3, av_dEdA3);
                add(50, dEdb1, av_dEdb1);
                add(100, dEdb2, av_dEdb2);
                add(10, dEdb3, av_dEdb3);
                //平均勾配を学習率に合わせて圧縮し、係数A,bを更新する
                scale(784 * 50, -eta / n, av_dEdA1);
                scale(50, -eta / n, av_dEdb1);
                scale(50 * 100, -eta / n, av_dEdA2);
                scale(100, -eta / n, av_dEdb2);
                scale(100 * 10, -eta / n, av_dEdA3);
                scale(10, -eta / n, av_dEdb3);
                add(784 * 50, av_dEdA1, A1);
                add(50 * 100, av_dEdA2, A2);
                add(100 * 10, av_dEdA3, A3);
                add(50, av_dEdb1, b1);
                add(100, av_dEdb2, b2);
                add(10, av_dEdb3, b3);
            }
        }
        //エポックごとに正答率と損失関数を表示する
        int accu = 0;
        float loss_sum = 0;
        int t;
        for (int j = 0; j < test_count; j++){
            t = inference6(A1, A2, A3, b1, b2, b3, test_x + j * 784, y);
            loss_sum += cross_entropy_error(y, test_y[j]);
            if(t==test_y[j]){
                accu++;
            }
        }
        printf("Epoch %3d  success=%f%%  loss=%f\n", i + 1, accu * 100.0 / test_count, loss_sum / test_count);
    }
    //コマンドライン引数で指定した3つのファイルに保存する
    save(argv[1], 50, 784, A1, b1);
    save(argv[2], 100, 50, A2, b2);
    save(argv[3], 10, 100, A3, b3);
    //メモリを開放する
    free(A1);
    free(A2);
    free(A3);
    free(b1);
    free(b2);
    free(b3);
    free(av_dEdA1);
    free(av_dEdA2);
    free(av_dEdA3);
    free(av_dEdb1);
    free(av_dEdb2);
    free(av_dEdb3);
    free(dEdA1);
    free(dEdA2);
    free(dEdA3);
    free(dEdb1);
    free(dEdb2);
    free(dEdb3);

    return 0;
}
