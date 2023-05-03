#include"nn.h"
#include<time.h>

//float型の配列oをxの値で初期化する関数
void init(int n, float x, float*o){
    int r;
    for (r = 0; r <= n - 1;r++){
        o[r] = x;
    }
}

//float型の配列A,x,oをに対し、掛け算を関数（Aはm×n行列）
void mul(int m, int n, const float*x,const float*A, float*o){
    int r;
    int l;
    for (l = 0; l <= m - 1; l++){
        for (r = l*n; r <= n-1+l*n; r++){
            o[l] = o[l] + x[r % n] * A[r];
        }
    }
}

//m×n行列A，m行列ベクトルb，n行列ベクトルxに対してy=Ax+bを行う関数
void fc(int m, int n, const float *x, const float *A, const float *b, float *o){
    int r;
    init(m, 0, o);
    mul(m, n, x, A, o);
    for (r = 0; r <= m - 1; r++){
        o[r] = o[r] + b[r];
    }
}

//n行列ベクトルxに対して、y[i]=x[i](x[i]>0),y[i]=0(x[i]<=0)(i=0,1,2,･･･)(Relu計算)を実行し、同じくn行列ベクトルのyに代入する関数
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
    float x_Max = x[0];
    for (i = 0; i < n; i++){
        if(x_Max < x[i]) {
            x_Max = x[i];
        }
    }
    return x_Max;
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

//6層NNによる推論でinputーfc1ーrelu1ーfc2ーrelu2ーfc3ーsoftmaxーoutputの順で計算され、0～9のいずれかの値を返す関数(推論)
int inference6(const float *A1,const float *A2,const float *A3,const float *b1,const float *b2,const float *b3,const float *x,float *y){
    float *y1 = malloc(sizeof(float) * 50);
    float *y2 = malloc(sizeof(float) * 100);
    fc(50,784,x,A1,b1,y1);   //fc1層
    relu(50, y1, y1);   //relu1層
    fc(100, 50, y1, A2, b2, y2);   //fc2層
    relu(100, y2, y2);   //relu2層
    fc(10, 100, y2, A3, b3, y);   //fc3層
    softmax(10, y, y);   //softmax層

    int n = 0;   //yが最大値の時の値 = ｛0，1，2，3，4，5，6，7，8，9｝のどれか
    float y_max = y[0];
    for (int a = 0; a < 10; a++){ //yの最大値を求める
        if(y[a]>y_max){
            y_max = y[a];
            n = a;
        }
    }
    //メモリ開放
    free(y1);
    free(y2);
    return n;
}

//もしファイルが開かなければ"FILE cannot open"と表示し、そうでなければデータを読み込む関数
void load(const char *filename, int m, int n, float *A, float *b){
    FILE *fp;
    if((fp=fopen(filename,"rb"))==NULL){
        printf("FILE cannot open\n");
    }else{
        fread(A, sizeof(float), m * n, fp);
        fread(b, sizeof(float), m, fp);
        fclose(fp);
    }
}

int main(int argc, char * argv[]){
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b1 = malloc(sizeof(float) * 50);
    float *b2 = malloc(sizeof(float) * 100);
    float *b3 = malloc(sizeof(float) * 10);
    float *y = malloc(sizeof(float) * 10);
    srand(time(NULL));
    //NN6にて保存したファイルの値を読み込む
    load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);

    //テスト用画像からランダムに画像を選び抜き、保存してから読み込む
    int i = rand() % (test_count + 1);
    save_mnist_bmp(test_x + 784 * i, argv[4], i);
    float *x = load_mnist_bmp(argv[4]);

    //｢0～9｣の値で、推論の結果と正解を表示する
    printf("inference %d   answer %d", inference6(A1, A2, A3, b1, b2, b3, x, y), test_y[i]);

    //推論結果と正解が一致した場合、　正解かどうかを表示する。
    if(inference6(A1, A2, A3, b1, b2, b3, x, y)==test_y[i]){
        printf(" Correct.\n");
    }else{
        printf(" The inference is incorrect.\n");
    }
    return 0;
}