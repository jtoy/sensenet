sudo apt-get update 
sudo apt-get install -y virtualbox-guest-dkms virtualbox-guest-utils virtualbox-guest-x11
sudo apt-get install -y python3 python3-pip
sudo pip3 install --upgrade pip
sudo apt-get install -y build-essential libfreetype6-dev libpng12-dev pkg-config python3-tk
sudo apt-get install -y binutils-gold freeglut3-dev libglew-dev mesa-common-dev libglm-dev
sudo ln -s /usr/include/freetype2/ft2build.h /usr/include/ft2build.h
sudo wget https://launchpad.net/ubuntu/+archive/primary/+files/libqhull7_2015.2-2_amd64.deb && sudo dpkg -i libqhull7_2015.2-2_amd64.deb
sudo wget https://launchpad.net/ubuntu/+archive/primary/+files/libqhull-dev_2015.2-2_amd64.deb && sudo dpkg -i libqhull-dev_2015.2-2_amd64.deb
sudo pip3 install numpy pybullet matplotlib 
sudo pip3 install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
sudo pip3 install torchvision
###########################################################
apt-get install -y xfce4
apt-get install -y gnome-icon-theme-full
apt-get install -y xserver-xorg-legacy xorg dbus-x11
dpkg-reconfigure xserver-xorg-legacy &> /dev/null
#############################################################
# Configuring X Org server to allow to be used by any users
#############################################################
sed -i 's/allowed_users=.*$/allowed_users=anybody/' /etc/X11/Xwrapper.config
echo "needs_root_rights=yes" >> /etc/X11/Xwrapper.config
#################################
# Enabling XFCE session on Boot 
#################################
cp /vagrant/provision/services/xfce.service /etc/systemd/system
chown root:root /etc/systemd/system/xfce.service
chmod 644 /etc/systemd/system/xfce.service
systemctl daemon-reload
systemctl enable xfce.service
systemctl start xfce &> /dev/null
