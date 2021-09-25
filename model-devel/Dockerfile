FROM ubuntu:latest


# 環境パスの設定
ENV PATH /usr/local/bin:$PATH


# 必要なもののインストール
# zsh, vim, git
RUN apt update
RUN apt install -y zsh git wget vim sudo


# zshの実行
RUN zsh
ENV SHELL /usr/bin/zsh


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
# RUN chown -R $USER /opt/conda
RUN conda create -n voiceappEnv python=3.8
SHELL ["conda", "run", "-n", "voiceappEnv", "/usr/bin/zsh", "-c"]
RUN conda install numpy matplotlib tqdm scikit-learn scipy pandas pillow jupyter
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN pip install torchinfo
RUN conda activate voiceappEnv
