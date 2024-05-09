create database db_mnist_exp;
use db_mnist_exp;

create table user (
    user_id int primary key auto_increment,
    name varchar(20) not null,
    password text not null,
    authority tinyint(1)
);
