# [OSSP] AWS EC2 Ubuntu 18.04

### | MySQL 

```bash
$ (sudo) wget https://dev.mysql.com/get/mysql-apt-config_0.8.13-1_all.deb

$ (sudo) dpkg -i mysql-apt-config_0.8.13-1_all.deb

$ (sudo) apt-get update

$ (sudo) apt-get install mysql-server

$ mysql_secure_installation

# Disallow root login remotely?  [YES]

# root 계정으로 mysql-server에 접속하지 못하도록 설정 
# 특정 테이블에 접근 및 수정 권한이 있는 유저 생성 후 공유 

# username = ossp-for-local 
# password = **************

$ CREATE USER 'ossp-for-local'@'%' identified by 'password'; 

$ GRANT ALL PRIVILEGES ON ossp_test.* TO 'ossp-for-local'@'%'; 

# 생성된 유저 확인
$ use mysql; 

$ SELECT host, user, authentication_string from users; 
```

### | Redis 

```bash
$ (sudo) apt update

$ (sudo) apt install redis-server

# Binding to localhost : 
Redis 설치 후 기본 설정으로 로컬에서만 접속할 수 있도록 설정되어 있다. 외부에서 접속하기 위해서는 우선 로컬에 바인딩된 설정을 외부 IP도 허용할 수 있도록 수정해야 한다. 또한, 설치된 Redis 서버가 AWS 인스턴스에 위치하므로, AWS 인바운드 보안 규칙에 Redis 서버 접근 포트 역시 허용해야 한다. 또한 보안 관련해서 비밀번호 설정한다. 

# 접근 IP 바인딩 정보가 위치한 파일 (-> 이 곳에서 바인딩될 IP 정보와 비밀번호 설정)
# bind -> 0.0.0.0 ::1 
# requirepass -> 비밀번호 
$ vi /etc/redis/redis.conf 

# 바인딩 조건이 바뀌었는 지 확인
$ (sudo) netstat -lnp | grep redis 

```



