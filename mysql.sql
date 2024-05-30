create database db_mnist_exp;
use db_mnist_exp;

create table user (
    user_id int primary key auto_increment,
    name varchar(20) not null,
    password text not null,
    authority tinyint(1)
);

# be sure there is at least one user!
# Password is '123456'
insert into user values (0, 'admin', '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', 1);