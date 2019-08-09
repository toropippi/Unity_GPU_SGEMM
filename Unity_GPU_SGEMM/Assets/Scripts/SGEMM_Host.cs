using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class SGEMM_Host : MonoBehaviour
{
    public ComputeShader shader;
    int kernelSGEMM_a;
    int kernelSGEMM_k;
    int kernelSGEMM_s;
    int kernelTrans;

    float[,] host_A;
    float[,] host_B;
    float[,] host_C;
    ComputeBuffer gpu_A;
    ComputeBuffer gpu_B;
    ComputeBuffer gpu_C;

    int n, m, k;

    void Start()
    {
        //配列 A は m x k 行列、配列 B は k x n 行列、配列 C は m x n 行列
        n = UnityEngine.Random.Range(1, 8193);
        m = UnityEngine.Random.Range(1, 8193);
        k = UnityEngine.Random.Range(1, 8193);
        Debug.Log("n=" + n + " m=" + m + " k=" + k + "");

        //カーネル設定
        kernelSGEMM_a = shader.FindKernel("SGEMM_a");
        kernelSGEMM_k = shader.FindKernel("SGEMM_k");
        kernelSGEMM_s = shader.FindKernel("SGEMM_small");
        kernelTrans = shader.FindKernel("Trans");

        //行列初期化
        HostandDeviseINIT(n, m, k);

        // host to device
        gpu_A.SetData(host_A);
        gpu_B.SetData(host_B);

        // GPUで計算
        mySGEMM(m, n, k, gpu_A, gpu_B, gpu_C);

        // device to host
        gpu_C.GetData(host_C);

        //検算
        CalcCheck();

        //次に、余談ではあるが転置行列の計算も
        MatrixTrans();

        //解放
        gpu_A.Release();
        gpu_B.Release();
        gpu_C.Release();
    }


    // Update is called once per frame
    void Update()
    {

    }


    void mySGEMM(int m, int n, int k, ComputeBuffer vram_A, ComputeBuffer vram_B, ComputeBuffer vram_C)
    {
        int kernel;
        if ((n < 128) | (m < 128))
        {
            kernel = kernelSGEMM_s;
        }
        else
        {
            if (k % 16 == 0)
            {
                kernel = kernelSGEMM_k;
            }
            else
            {
                kernel = kernelSGEMM_a;
            }
        }
        //引数をセット
        shader.SetInt("N", n);
        shader.SetInt("M", m);
        shader.SetInt("K", k);
        shader.SetBuffer(kernel, "A", vram_A);
        shader.SetBuffer(kernel, "B", vram_B);
        shader.SetBuffer(kernel, "C", vram_C);
        shader.Dispatch(kernel, (n + 127) / 128, (m + 127) / 128, 1);
    }


    void HostandDeviseINIT(int n, int m, int k)
    {
        host_A = new float[m, k];
        host_B = new float[k, n];
        host_C = new float[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                host_A[i, j] = UnityEngine.Random.Range(0.0f, 1.0f);
            }
        }
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                host_B[i, j] = UnityEngine.Random.Range(0.0f, 1.0f);
            }
        }

        gpu_A = new ComputeBuffer(m * k, sizeof(float));
        gpu_B = new ComputeBuffer(k * n, sizeof(float));
        gpu_C = new ComputeBuffer(m * n, sizeof(float));
    }


    void CalcCheck()
    {
        //host_C[mm, nn]が正しいか確認
        int mm = UnityEngine.Random.Range(0, m);
        int nn = UnityEngine.Random.Range(0, n);
        float ans = host_C[mm, nn];
        float checksum = 0.0f;
        for (int kk = 0; kk < k; kk++)
        {
            checksum += host_A[mm, kk] * host_B[kk, nn];
        }

        Debug.Log("デバッグ C[" + mm + "," + nn + "] =");
        Debug.Log("GPUの結果" + ans);
        Debug.Log("CPUの結果" + checksum);
    }

    void MatrixTrans()
    {
        Debug.Log("");
        gpu_C.GetData(host_C);
        int mm = UnityEngine.Random.Range(0, m);
        int nn = UnityEngine.Random.Range(0, n);
        Debug.Log("転置の前後C[" + mm + "," + nn + "] =");
        Debug.Log("前"+ host_C[mm, nn]);

        ComputeBuffer gpu_CT = new ComputeBuffer(n * m, sizeof(float));
        shader.SetInt("N", n);
        shader.SetInt("M", m);
        shader.SetBuffer(kernelTrans, "A", gpu_C);
        shader.SetBuffer(kernelTrans, "AT", gpu_CT);
        shader.Dispatch(kernelTrans, (n + 15) / 16, (m + 15) / 16, 1);

        host_C = new float[n, m];
        gpu_CT.GetData(host_C);

        float anst = host_C[nn, mm];
        Debug.Log("後"+ host_C[nn, mm]);

        gpu_CT.Release();
    }
}