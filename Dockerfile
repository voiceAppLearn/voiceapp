FROM ubuntu:latest


# 必要なもののインストール
# zsh, vim, git
RUN apt-get update
RUN apt-get install -y zsh git wget


# zshの実行
RUN zsh
ENV SHELL /usr/bin/zsh


# 環境パスの設定
ENV PATH /usr/local/bin:$PATH
# ユーザ，ホームパスの決定
ENV USER VoiceApp
ENV HOME /home/$USER
# ユーザーの追加
RUN useradd -m $USER
RUN gpasswd -a $USER sudo
RUN echo "${USER}:VoiceApp_pass" | chpasswd
# 以降のコマンドを実行するユーザーを決める
USER $USER


# ホームディレクトリへ移動
WORKDIR $HOME
# 設定ファイルの読み込み
RUN git clone https://github.com/mlplab/dotfiles_mlplab
RUN mkdir .vim
RUN sh dotfiles_mlplab/dotfilesLink.sh
# Miniconda の install
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN zsh ./Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH $PATH:$HOME/miniconda3/bin


# 環境設定
RUN conda create -n voiceappEnv python=3.8
SHELL ["conda", "run", "-n", "voiceappEnv", "/usr/bin/zsh", "-c"]
RUN conda install numpy matplotlib tqdm scikit-learn scipy pandas pillow
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN pip install torchinfo
