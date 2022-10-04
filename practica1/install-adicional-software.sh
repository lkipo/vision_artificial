###############################################################
# INSTALAMOS SOFTWARE ADICONAL EN UBUNTU                    #
###############################################################

# | SCRIP COMPROBADO SOBRE O SO          |
# |--------------------------------------|
# | SO             | Test | Ultimo test  |
# |----------------|------|--------------|
# | Ubuntu 20.04   | OK   | 12/11/2020   |


# 1.ACTUALIZAMOS UBUNTU
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove


# 2. SOFTWARE HABITUAL

sudo apt-get install -y build-essential cmake               # FERRAMENTAS DE DESENVOLVEMENTO
sudo apt-get install -y p7zip p7zip-full unrar-free unzip   # COMPRESORES
sudo apt-get install -y htop lshw wget locate               # UTILIDADES 
sudo apt-get install -y screen                              # TERMINAL MULTIPLEXADA
sudo apt-get install -y nano vim                            # EDITORES DE TEXTO PARA SO
sudo apt-get install -y git subversion                      # REPOSITORIOS
sudo apt-get install -y gdisk                               # FERRAMENTA DE PARTICION DISCO
sudo apt-get install -y pdftk                               # MANIPULACION PDF
sudo apt-get install -y ffmpeg                              # VIDEO
sudo apt-get install -y default-jdk                         # JAVA DEVELOPMENT KIT (JDK)
sudo apt-get install -y wavemon                             # REDE
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng     # OCR


# 3. GUI SOFTWARE

sudo apt-get install -y gparted                             # PARTICION DE DISCO
sudo apt-get install -y network-manager-openvpn-gnome       # VPN
sudo apt-get install -y network-manager-openvpn             # OPENVPN
sudo apt-get install -y transmission-gtk                    # CLIENTE BITTORRENT
sudo apt-get install -y galculator                          # CALCULADORA CIENTÍFICA
sudo apt-get install -y vlc                                 # VIDEO E PODCAST 
sudo apt-get install -y pavucontrol                         # CONTROL DE VOLUME

sudo apt-get install -y geany                               # EDITOR SIMPLE DE TEXTO
sudo apt-get install -y blender gimp imagemagick inkscape   # EDITORES GRAFICOS
sudo apt-get install -y audacity                            # EDITOR DE AUDIO
sudo apt-get install -y openshot                            # EDITOR DE VIDEO
sudo apt-get install -y filezilla                           # CLIENTE FTP/FTPS/SFTP
# sudo apt-get install -y libreoffice                       # OFFICE 
sudo apt-get install -y texlive-full texstudio              # LATEX (o paquete mais completo!)
# sudo apt-get install -y kazam obs-studio                  # SCREENCAST 
