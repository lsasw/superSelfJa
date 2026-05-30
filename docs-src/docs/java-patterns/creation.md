---
title: 创建型设计模式
icon: puzzle-piece
order: 1
---

# Java 创建型设计模式详解

创建型设计模式关注的是对象的创建过程，旨在提高系统创建对象的灵活性和可复用性。本文档详细介绍 Java 中最常用的五种创建型设计模式。

---

## 一、单例模式（Singleton Pattern）

### 1.1 模式定义

单例模式确保一个类只有一个实例，并提供全局访问点。这是最简单但也最容易被错误实现的设计模式。

### 1.2 应用场景

- 配置管理器：整个应用只需要一个配置实例
- 线程池：避免创建多个线程池导致资源浪费
- 数据库连接池：控制数据库连接数量
- 日志记录器：统一的日志输出入口
- 缓存管理器：全局共享的缓存实例

### 1.3 实现方式

#### 方式一：饿汉式（线程安全）

```java
public class Singleton {
    // 类加载时即创建实例
    private static final Singleton INSTANCE = new Singleton();
    
    private Singleton() {
        // 私有构造函数
    }
    
    public static Singleton getInstance() {
        return INSTANCE;
    }
}
```

**优点**：线程安全，实现简单  
**缺点**：类加载时就创建实例，可能造成资源浪费

#### 方式二：懒汉式（双重检查锁定）

```java
public class Singleton {
    // volatile 保证多线程下的可见性
    private static volatile Singleton instance;
    
    private Singleton() {}
    
    public static Singleton getInstance() {
        // 第一次检查：避免不必要的同步
        if (instance == null) {
            synchronized (Singleton.class) {
                // 第二次检查：确保只创建一个实例
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**关键点**：
- `volatile` 关键字防止指令重排序
- 双重检查减少锁的开销
- 仅在第一次创建时同步

#### 方式三：静态内部类（推荐）

```java
public class Singleton {
    private Singleton() {}
    
    // 静态内部类，只有在调用 getInstance 时才加载
    private static class Holder {
        static final Singleton INSTANCE = new Singleton();
    }
    
    public static Singleton getInstance() {
        return Holder.INSTANCE;
    }
}
```

**优点**：
- 利用类加载机制保证线程安全
- 延迟加载，节省资源
- 代码简洁

#### 方式四：枚举单例（最有效）

```java
public enum Singleton {
    INSTANCE;
    
    public void doSomething() {
        // 业务方法
    }
}
```

**优点**：
- 绝对防止反射和序列化破坏单例
- 代码最简洁
- Joshua Bloch 在《Effective Java》中推荐

### 1.4 Spring 中的单例

Spring 框架默认使用单例模式管理 Bean：

```java
@Configuration
public class AppConfig {
    @Bean
    @Scope("singleton") // 默认就是 singleton
    public DataSource dataSource() {
        return new DataSource();
    }
}
```

---

## 二、工厂方法模式（Factory Method Pattern）

### 2.1 模式定义

定义一个用于创建对象的接口，让子类决定实例化哪一个类。工厂方法使一个类的实例化延迟到其子类。

### 2.2 应用场景

- 日志记录器：根据不同环境创建文件日志、控制台日志
- 数据库连接：根据配置创建 MySQL、Oracle 连接
- 支付系统：根据支付方式创建不同的支付处理器
- UI 组件：根据平台创建不同的按钮、文本框

### 2.3 实现示例

#### 支付系统实战

```java
// 产品接口
public interface PaymentProcessor {
    void process(double amount);
}

// 具体产品
public class AlipayProcessor implements PaymentProcessor {
    @Override
    public void process(double amount) {
        System.out.println("支付宝支付: ¥" + amount);
    }
}

public class WechatPayProcessor implements PaymentProcessor {
    @Override
    public void process(double amount) {
        System.out.println("微信支付: ¥" + amount);
    }
}

public class BankCardProcessor implements PaymentProcessor {
    @Override
    public void process(double amount) {
        System.out.println("银行卡支付: ¥" + amount);
    }
}

// 工厂接口
public interface PaymentFactory {
    PaymentProcessor createProcessor();
}

// 具体工厂
public class AlipayFactory implements PaymentFactory {
    @Override
    public PaymentProcessor createProcessor() {
        return new AlipayProcessor();
    }
}

public class WechatPayFactory implements PaymentFactory {
    @Override
    public PaymentProcessor createProcessor() {
        return new WechatPayProcessor();
    }
}

// 使用示例
public class PaymentService {
    public void pay(PaymentFactory factory, double amount) {
        PaymentProcessor processor = factory.createProcessor();
        processor.process(amount);
    }
}
```

### 2.4 Spring 中的工厂模式

Spring 的 `BeanFactory` 和 `ApplicationContext` 就是工厂模式的典型应用：

```java
// 获取 Bean 的工厂方法
ApplicationContext context = new ClassPathXmlApplicationContext("beans.xml");
UserService userService = context.getBean("userService", UserService.class);
```

---

## 三、抽象工厂模式（Abstract Factory Pattern）

### 3.1 模式定义

提供一个创建一系列相关或相互依赖对象的接口，而无须指定它们具体的类。

### 3.2 与工厂方法的区别

| 对比项 | 工厂方法 | 抽象工厂 |
|--------|----------|----------|
| 产品数量 | 单个产品 | 产品族 |
| 扩展性 | 易扩展新产品 | 易扩展新工厂 |
| 复杂度 | 较低 | 较高 |

### 3.3 UI 组件工厂实战

```java
// 抽象产品族
public interface Button {
    void render();
}

public interface Checkbox {
    void check();
}

// 具体产品族 - Windows
public class WindowsButton implements Button {
    @Override
    public void render() {
        System.out.println("渲染 Windows 风格按钮");
    }
}

public class WindowsCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("勾选 Windows 风格复选框");
    }
}

// 具体产品族 - macOS
public class MacButton implements Button {
    @Override
    public void render() {
        System.out.println("渲染 macOS 风格按钮");
    }
}

public class MacCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("勾选 macOS 风格复选框");
    }
}

// 抽象工厂
public interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
}

// 具体工厂
public class WindowsFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }
    
    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();
    }
}

public class MacFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new MacButton();
    }
    
    @Override
    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}

// 客户端代码
public class Application {
    private Button button;
    private Checkbox checkbox;
    
    public Application(GUIFactory factory) {
        this.button = factory.createButton();
        this.checkbox = factory.createCheckbox();
    }
    
    public void render() {
        button.render();
        checkbox.check();
    }
}
```

---

## 四、建造者模式（Builder Pattern）

### 4.1 模式定义

将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示。

### 4.2 应用场景

- 构建复杂对象：如 HTTP 请求、SQL 查询
- 不可变对象：需要设置多个属性的对象
- 配置对象：具有大量可选参数的配置类

### 4.3 Java 标准库中的建造者模式

```java
// StringBuilder 是建造者模式的典型应用
StringBuilder sb = new StringBuilder();
sb.append("Hello")
  .append(" ")
  .append("World")
  .append("!");
String result = sb.toString();
```

### 4.4 实战：用户对象构建

```java
public class User {
    // 必需参数
    private final String username;
    private final String email;
    
    // 可选参数
    private final String nickname;
    private final int age;
    private final String phone;
    private final String avatar;
    private final String address;
    
    // 私有构造函数
    private User(Builder builder) {
        this.username = builder.username;
        this.email = builder.email;
        this.nickname = builder.nickname;
        this.age = builder.age;
        this.phone = builder.phone;
        this.avatar = builder.avatar;
        this.address = builder.address;
    }
    
    public static Builder builder(String username, String email) {
        return new Builder(username, email);
    }
    
    // Getter 方法...
    
    // 建造者内部类
    public static class Builder {
        // 必需参数
        private final String username;
        private final String email;
        
        // 可选参数
        private String nickname;
        private int age;
        private String phone;
        private String avatar;
        private String address;
        
        private Builder(String username, String email) {
            this.username = username;
            this.email = email;
        }
        
        public Builder nickname(String nickname) {
            this.nickname = nickname;
            return this;
        }
        
        public Builder age(int age) {
            this.age = age;
            return this;
        }
        
        public Builder phone(String phone) {
            this.phone = phone;
            return this;
        }
        
        public Builder avatar(String avatar) {
            this.avatar = avatar;
            return this;
        }
        
        public Builder address(String address) {
            this.address = address;
            return this;
        }
        
        public User build() {
            // 可以在这里添加验证逻辑
            if (username == null || username.isEmpty()) {
                throw new IllegalArgumentException("用户名不能为空");
            }
            if (email == null || !email.contains("@")) {
                throw new IllegalArgumentException("邮箱格式不正确");
            }
            return new User(this);
        }
    }
}

// 使用示例
User user = User.builder("zhangsan", "zhangsan@example.com")
    .nickname("张三")
    .age(28)
    .phone("13800138000")
    .avatar("avatar.jpg")
    .address("北京市朝阳区")
    .build();
```

### 4.5 Lombok 简化建造者

```java
@Builder
@Data
public class Product {
    private String name;
    private double price;
    private String category;
    private String description;
}

// 使用
Product product = Product.builder()
    .name("Java 编程思想")
    .price(99.9)
    .category("图书")
    .description("经典 Java 入门书籍")
    .build();
```

---

## 五、原型模式（Prototype Pattern）

### 5.1 模式定义

用原型实例指定创建对象的种类，并且通过拷贝这些原型创建新的对象。

### 5.2 Java 中的原型模式

```java
public class Document implements Cloneable {
    private String title;
    private String content;
    private List<String> tags;
    private Date createTime;
    
    // 浅拷贝
    @Override
    public Document clone() {
        try {
            return (Document) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
    
    // 深拷贝
    public Document deepClone() {
        try {
            Document cloned = (Document) super.clone();
            cloned.tags = new ArrayList<>(this.tags);
            cloned.createTime = (Date) this.createTime.clone();
            return cloned;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
```

### 5.3 浅拷贝 vs 深拷贝

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| 浅拷贝 | 只复制基本类型，引用类型共享 | 对象不包含可变引用 |
| 深拷贝 | 递归复制所有引用对象 | 需要完全独立的副本 |

---

## 六、模式选择指南

| 需求 | 推荐模式 |
|------|----------|
| 确保只有一个实例 | 单例模式 |
| 创建单一类型的对象，但具体类型未知 | 工厂方法 |
| 创建一族相关对象 | 抽象工厂 |
| 对象构造过程复杂，有多个可选参数 | 建造者模式 |
| 需要快速复制现有对象 | 原型模式 |

---

## 七、实战建议

1. **不要过度使用**：简单的对象直接 `new` 即可
2. **结合使用**：工厂 + 单例是常见组合
3. **Spring 优先**：在 Spring 项目中，优先使用 Spring 的 IoC 容器
4. **测试覆盖**：工厂和建造者需要充分的单元测试
5. **文档注释**：为工厂方法添加清晰的 JavaDoc

---

> 💡 **提示**：设计模式不是银弹，理解模式背后的思想比死记硬背更重要。在实际开发中，应根据具体场景灵活选择。
