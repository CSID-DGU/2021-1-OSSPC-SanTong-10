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





