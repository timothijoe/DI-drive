动机：
汽车能学会减速等待，也能学会加速超车；
但是变道的动作过于频繁，而且角度不是很稳定，可能这才是汽车无法变道稳定的原因

因此
a1:dim=10, auto_alpha， extra_heading + another lateral
a2:dim=10, fixed_alpha, extra_heading + another lateral
a3:dim=10, auto_alpha, crash_vehicle = 2.0 降低碰撞惩罚, extra_heading
a4:dim=10, fixed_ahlpha, crash_vehicle = 2.0 降低碰撞惩罚  extra_heading

文件夹：
sp5_dim10_gama09_auto_extra
sp6_dim10_gama09_extra
sp7_dim10_gama09_auto_lc_extra
sp8_dim10_gama09_lc_extra
