动机：
gamma = 0.9的时候，参数效果可以参考；
auto_alpha好于 fixed_alpha
dim = 10 好于 dim= 3

因此
a1:dim=10, auto_alpha
a2:dim=10, fixed_alpha
a3:dim=10, auto_alpha, crash_vehicle = 2.0 降低碰撞惩罚
a4:dim=10, fixed_ahlpha, crash_vehicle = 2.0 降低碰撞惩罚

文件夹：
sp1_dim10_gama09_auto
sp2_dim10_gama09
sp3_dim10_gama09_auto_lc
sp4_dim10_gama09_lc
