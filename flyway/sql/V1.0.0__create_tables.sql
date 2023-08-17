CREATE TABLE IF NOT EXISTS tenants (
  id serial PRIMARY KEY,
  code varchar(12) NOT NULL UNIQUE,
  name varchar(50) NOT NULL UNIQUE,
  description varchar(1000)
);

-- users
CREATE TABLE IF NOT EXISTS users (
  id serial PRIMARY KEY,
  tenant_id integer NOT NULL REFERENCES tenants (id),
  name varchar(50) NOT NULL,
  mail_address varchar(100) NOT NULL,
  UNIQUE(tenant_id, mail_address)
);

-- user profile
CREATE TABLE IF NOT EXISTS user_profile (
  id serial PRIMARY KEY,
  user_id integer NOT NULL REFERENCES users (id) UNIQUE,
  description varchar(1000)
);

-- user career history
CREATE TABLE IF NOT EXISTS user_career_histories (
  id serial PRIMARY KEY,
  user_id integer NOT NULL REFERENCES users (id),
  company_name varchar(50) NOT NULL,
  position varchar(50),
  description varchar(1000),
  UNIQUE(user_id, company_name)
);
