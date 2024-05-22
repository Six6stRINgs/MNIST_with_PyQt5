create database db_mnist_exp;
use db_mnist_exp;

create table user (
    user_id int primary key auto_increment,
    name varchar(20) not null,
    password text not null,
    authority tinyint(1)
);

# be sure there is at least one admin user!
insert into user values (0, 'admin', '123456', 1);