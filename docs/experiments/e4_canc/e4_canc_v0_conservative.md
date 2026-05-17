# E4-CANC-v0: Conservative Conflict-Aware Negative Cache

这一步会真正修改 negative cache 的准入规则：

原始 Point-Cache：

Inegbase(x)=IH(x)I_{neg}^{base}(x)=I_H(x)Inegbase(x)=IH(x)

改成：

InegE4(x)=IH(x)∨(IM(x)∧ID(x))I_{neg}^{E4}(x)=I_H(x)\lor(I_M(x)\land I_D(x))InegE4(x)=IH(x)∨(IM(x)∧ID(x))

其中：

ID(x)=1[Dglmean(x)>τd]I_D(x)=\mathbf{1}[D_{gl}^{mean}(x)>\tau_d]ID(x)=1[Dglmean(x)>τd]

第一版固定：



```
tau_d = 5e-5
tau_m = 0.20
tau_p = 0.40
```



注意：**仍然不使用 local alternative class 作为 pseudo-label**，只把 global predicted class y^g\hat y_gy^g 放进 negative cache 进行 suppression。