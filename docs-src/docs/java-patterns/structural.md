---
title: 结构型设计模式
icon: puzzle-piece
order: 2
---

# Java 结构型设计模式详解

结构型设计模式关注的是如何组合类和对象以形成更大的结构，从而提高系统的灵活性和可维护性。

---

## 一、代理模式（Proxy Pattern）

### 1.1 模式定义

为其他对象提供一种代理以控制对这个对象的访问。代理对象可以在客户端和目标对象之间起到中介作用。

### 1.2 应用场景

- 远程代理：访问远程服务
- 虚拟代理：延迟加载大对象
- 保护代理：控制访问权限
- 缓存代理：缓存计算结果
- AOP 代理：Spring AOP 的核心实现

### 1.3 静态代理实现

```java
// 接口
public interface UserService {
    void createUser(String name);
    User getUserById(Long id);
}

// 真实对象
public class UserServiceImpl implements UserService {
    @Override
    public void createUser(String name) {
        System.out.println("创建用户: " + name);
    }
    
    @Override
    public User getUserById(Long id) {
        System.out.println("查询用户: " + id);
        return new User(id, "张三");
    }
}

// 代理对象
public class UserServiceProxy implements UserService {
    private UserService target;
    
    public UserServiceProxy(UserService target) {
        this.target = target;
    }
    
    @Override
    public void createUser(String name) {
        System.out.println("[代理] 权限检查...");
        System.out.println("[代理] 记录日志...");
        target.createUser(name);
        System.out.println("[代理] 事务提交");
    }
    
    @Override
    public User getUserById(Long id) {
        System.out.println("[代理] 缓存检查...");
        return target.getUserById(id);
    }
}
```

### 1.4 动态代理（JDK）

```java
public class JdkProxyFactory {
    
    @SuppressWarnings("unchecked")
    public static <T> T createProxy(T target, Object handler) {
        return (T) Proxy.newProxyInstance(
            target.getClass().getClassLoader(),
            target.getClass().getInterfaces(),
            new InvocationHandler() {
                @Override
                public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                    System.out.println("[JDK 代理] 前置处理: " + method.getName());
                    long start = System.currentTimeMillis();
                    
                    Object result = method.invoke(target, args);
                    
                    long cost = System.currentTimeMillis() - start;
                    System.out.println("[JDK 代理] 后置处理，耗时: " + cost + "ms");
                    return result;
                }
            }
        );
    }
}

// 使用示例
UserService service = new UserServiceImpl();
UserService proxy = JdkProxyFactory.createProxy(service, null);
proxy.createUser("李四");
```

### 1.5 CGLIB 代理

```java
public class CglibProxyFactory implements MethodInterceptor {
    
    @SuppressWarnings("unchecked")
    public static <T> T createProxy(Class<T> targetClass) {
        Enhancer enhancer = new Enhancer();
        enhancer.setSuperclass(targetClass);
        enhancer.setCallback(new CglibProxyFactory());
        return (T) enhancer.create();
    }
    
    @Override
    public Object intercept(Object obj, Method method, Object[] args, MethodProxy proxy) throws Throwable {
        System.out.println("[CGLIB 代理] 前置处理");
        Object result = proxy.invokeSuper(obj, args);
        System.out.println("[CGLIB 代理] 后置处理");
        return result;
    }
}
```

### 1.6 Spring AOP 中的代理

```java
@Aspect
@Component
public class LogAspect {
    
    @Before("execution(* com.example.service.*.*(..))")
    public void beforeMethod(JoinPoint joinPoint) {
        System.out.println("方法执行前: " + joinPoint.getSignature().getName());
    }
    
    @After("execution(* com.example.service.*.*(..))")
    public void afterMethod(JoinPoint joinPoint) {
        System.out.println("方法执行后");
    }
    
    @Around("execution(* com.example.service.*.*(..))")
    public Object aroundMethod(ProceedingJoinPoint joinPoint) throws Throwable {
        long start = System.currentTimeMillis();
        Object result = joinPoint.proceed();
        long cost = System.currentTimeMillis() - start;
        System.out.println("方法耗时: " + cost + "ms");
        return result;
    }
}
```

---

## 二、适配器模式（Adapter Pattern）

### 2.1 模式定义

将一个类的接口转换成客户希望的另外一个接口，使原本由于接口不兼容而不能一起工作的那些类可以一起工作。

### 2.2 应用场景

- 新旧系统对接
- 第三方库集成
- 统一接口规范

### 2.3 类适配器

```java
// 已存在的类（不兼容的接口）
public class LegacyPayment {
    public void oldPay(double amount) {
        System.out.println("旧版支付: " + amount);
    }
}

// 目标接口
public interface Payment {
    void pay(double amount);
}

// 类适配器（继承 + 实现）
public class PaymentAdapter extends LegacyPayment implements Payment {
    @Override
    public void pay(double amount) {
        oldPay(amount);
    }
}
```

### 2.4 对象适配器（推荐）

```java
public class PaymentAdapter implements Payment {
    private LegacyPayment adaptee;
    
    public PaymentAdapter(LegacyPayment adaptee) {
        this.adaptee = adaptee;
    }
    
    @Override
    public void pay(double amount) {
        adaptee.oldPay(amount);
    }
}
```

### 2.5 Spring MVC 中的适配器

```java
// HandlerAdapter 是适配器模式的典型应用
public interface HandlerAdapter {
    boolean supports(Object handler);
    ModelAndView handle(HttpServletRequest request, 
                       HttpServletResponse response, 
                       Object handler) throws Exception;
}

// 不同类型的 Controller 有不同的 Adapter
public class SimpleControllerHandlerAdapter implements HandlerAdapter {
    @Override
    public boolean supports(Object handler) {
        return (handler instanceof Controller);
    }
    
    @Override
    public ModelAndView handle(HttpServletRequest request, 
                              HttpServletResponse response, 
                              Object handler) throws Exception {
        return ((Controller) handler).handleRequest(request, response);
    }
}
```

---

## 三、装饰器模式（Decorator Pattern）

### 3.1 模式定义

动态地给一个对象添加一些额外的职责，比生成子类更为灵活。

### 3.2 Java IO 中的装饰器

```java
// Java IO 是装饰器模式的经典应用
InputStream input = new FileInputStream("file.txt");
input = new BufferedInputStream(input);    // 添加缓冲功能
input = new GZIPInputStream(input);        // 添加解压缩功能

// 装饰器链：GZIPInputStream -> BufferedInputStream -> FileInputStream
```

### 3.3 实战：咖啡订单系统

```java
// 抽象组件
public abstract class Coffee {
    public abstract String getDescription();
    public abstract double getCost();
}

// 具体组件
public class Espresso extends Coffee {
    @Override
    public String getDescription() {
        return "浓缩咖啡";
    }
    
    @Override
    public double getCost() {
        return 20.0;
    }
}

// 装饰器抽象类
public abstract class CoffeeDecorator extends Coffee {
    protected Coffee decoratedCoffee;
    
    public CoffeeDecorator(Coffee coffee) {
        this.decoratedCoffee = coffee;
    }
    
    @Override
    public String getDescription() {
        return decoratedCoffee.getDescription();
    }
    
    @Override
    public double getCost() {
        return decoratedCoffee.getCost();
    }
}

// 具体装饰器 - 加奶
public class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }
    
    @Override
    public String getDescription() {
        return decoratedCoffee.getDescription() + " + 牛奶";
    }
    
    @Override
    public double getCost() {
        return decoratedCoffee.getCost() + 5.0;
    }
}

// 具体装饰器 - 加糖
public class SugarDecorator extends CoffeeDecorator {
    public SugarDecorator(Coffee coffee) {
        super(coffee);
    }
    
    @Override
    public String getDescription() {
        return decoratedCoffee.getDescription() + " + 糖";
    }
    
    @Override
    public double getCost() {
        return decoratedCoffee.getCost() + 3.0;
    }
}

// 使用示例
Coffee coffee = new Espresso();
coffee = new MilkDecorator(coffee);
coffee = new SugarDecorator(coffee);

System.out.println(coffee.getDescription()); // 浓缩咖啡 + 牛奶 + 糖
System.out.println("价格: ¥" + coffee.getCost()); // 28.0
```

### 3.4 Spring 中的装饰器

```java
// ServletRequestWrapper 是装饰器模式的实现
HttpServletRequest wrappedRequest = new HttpServletRequestWrapper(request) {
    @Override
    public String getParameter(String name) {
        // 添加 XSS 过滤
        String value = super.getParameter(name);
        return value != null ? XSSFilter.clean(value) : null;
    }
};
```

---

## 四、组合模式（Composite Pattern）

### 4.1 模式定义

将对象组合成树形结构以表示"部分-整体"的层次结构，使客户端对单个对象和组合对象的使用具有一致性。

### 4.2 文件系统实战

```java
// 抽象组件
public abstract class FileSystemNode {
    protected String name;
    
    public FileSystemNode(String name) {
        this.name = name;
    }
    
    public abstract void print(int depth);
    public abstract int getSize();
}

// 叶子节点 - 文件
public class File extends FileSystemNode {
    private int size;
    
    public File(String name, int size) {
        super(name);
        this.size = size;
    }
    
    @Override
    public void print(int depth) {
        String indent = "  ".repeat(depth);
        System.out.println(indent + " " + name + " (" + size + "KB)");
    }
    
    @Override
    public int getSize() {
        return size;
    }
}

// 组合节点 - 目录
public class Directory extends FileSystemNode {
    private List<FileSystemNode> children = new ArrayList<>();
    
    public Directory(String name) {
        super(name);
    }
    
    public void addChild(FileSystemNode node) {
        children.add(node);
    }
    
    @Override
    public void print(int depth) {
        String indent = "  ".repeat(depth);
        System.out.println(indent + "📁 " + name);
        for (FileSystemNode child : children) {
            child.print(depth + 1);
        }
    }
    
    @Override
    public int getSize() {
        return children.stream().mapToInt(FileSystemNode::getSize).sum();
    }
}

// 使用示例
Directory root = new Directory("/");
Directory docs = new Directory("docs");
docs.addChild(new File("README.md", 5));
docs.addChild(new File("guide.pdf", 120));

Directory src = new Directory("src");
src.addChild(new File("Main.java", 8));
src.addChild(new File("Utils.java", 15));

root.addChild(docs);
root.addChild(src);

root.print(0);
System.out.println("总大小: " + root.getSize() + "KB");
```

---

## 五、外观模式（Facade Pattern）

### 5.1 模式定义

为子系统中的一组接口提供一个一致的界面，此模式定义了一个高层接口，使得子系统更加容易使用。

### 5.2 实战：电商订单系统

```java
// 子系统 - 库存服务
public class InventoryService {
    public boolean checkStock(String productId, int quantity) {
        System.out.println("检查库存: " + productId);
        return true;
    }
    
    public void deductStock(String productId, int quantity) {
        System.out.println("扣减库存: " + productId + " x " + quantity);
    }
}

// 子系统 - 支付服务
public class PaymentService {
    public boolean processPayment(String userId, double amount) {
        System.out.println("处理支付: 用户 " + userId + ", 金额 " + amount);
        return true;
    }
}

// 子系统 - 物流服务
public class ShippingService {
    public void createShipping(String orderId, String address) {
        System.out.println("创建物流: 订单 " + orderId + ", 地址 " + address);
    }
}

// 子系统 - 通知服务
public class NotificationService {
    public void sendEmail(String email, String message) {
        System.out.println("发送邮件: " + email + ", 内容: " + message);
    }
}

// 外观类
public class OrderFacade {
    private InventoryService inventoryService = new InventoryService();
    private PaymentService paymentService = new PaymentService();
    private ShippingService shippingService = new ShippingService();
    private NotificationService notificationService = new NotificationService();
    
    public String createOrder(String userId, String productId, int quantity, 
                             double amount, String address, String email) {
        System.out.println("=== 开始创建订单 ===");
        
        // 1. 检查库存
        if (!inventoryService.checkStock(productId, quantity)) {
            throw new RuntimeException("库存不足");
        }
        
        // 2. 处理支付
        if (!paymentService.processPayment(userId, amount)) {
            throw new RuntimeException("支付失败");
        }
        
        // 3. 扣减库存
        inventoryService.deductStock(productId, quantity);
        
        // 4. 创建物流
        String orderId = generateOrderId();
        shippingService.createShipping(orderId, address);
        
        // 5. 发送通知
        notificationService.sendEmail(email, "订单创建成功: " + orderId);
        
        System.out.println("=== 订单创建完成 ===");
        return orderId;
    }
    
    private String generateOrderId() {
        return "ORD" + System.currentTimeMillis();
    }
}

// 客户端代码（大大简化）
public class OrderClient {
    public void placeOrder() {
        OrderFacade facade = new OrderFacade();
        String orderId = facade.createOrder(
            "user123", "PROD001", 2, 199.8, "北京市朝阳区", "user@example.com"
        );
        System.out.println("订单号: " + orderId);
    }
}
```

---

## 六、桥接模式（Bridge Pattern）

### 6.1 模式定义

将抽象部分与它的实现部分分离，使它们都可以独立地变化。

### 6.2 消息发送系统实战

```java
// 实现接口 - 消息发送方式
public interface MessageSender {
    void send(String message, String receiver);
}

// 具体实现 - 邮件发送
public class EmailSender implements MessageSender {
    @Override
    public void send(String message, String receiver) {
        System.out.println("发送邮件到: " + receiver);
        System.out.println("邮件内容: " + message);
    }
}

// 具体实现 - 短信发送
public class SmsSender implements MessageSender {
    @Override
    public void send(String message, String receiver) {
        System.out.println("发送短信到: " + receiver);
        System.out.println("短信内容: " + message);
    }
}

// 具体实现 - 站内信发送
public class InAppSender implements MessageSender {
    @Override
    public void send(String message, String receiver) {
        System.out.println("发送站内信到: " + receiver);
        System.out.println("消息内容: " + message);
    }
}

// 抽象类 - 消息类型
public abstract class Message {
    protected MessageSender sender;
    
    public Message(MessageSender sender) {
        this.sender = sender;
    }
    
    public abstract void sendMessage(String content, String receiver);
}

// 具体抽象 - 通知消息
public class NotificationMessage extends Message {
    public NotificationMessage(MessageSender sender) {
        super(sender);
    }
    
    @Override
    public void sendMessage(String content, String receiver) {
        String message = "【通知】" + content;
        sender.send(message, receiver);
    }
}

// 具体抽象 - 验证码消息
public class VerifyCodeMessage extends Message {
    public VerifyCodeMessage(MessageSender sender) {
        super(sender);
    }
    
    @Override
    public void sendMessage(String content, String receiver) {
        String message = "【验证码】" + content + "，5分钟内有效";
        sender.send(message, receiver);
    }
}

// 使用示例
MessageSender email = new EmailSender();
Message notification = new NotificationMessage(email);
notification.sendMessage("您有新订单", "user@example.com");

MessageSender sms = new SmsSender();
Message verifyCode = new VerifyCodeMessage(sms);
verifyCode.sendMessage("123456", "13800138000");
```

---

## 七、享元模式（Flyweight Pattern）

### 7.1 模式定义

运用共享技术有效地支持大量细粒度的对象。

### 7.2 实战：数据库连接池

```java
public class ConnectionPool {
    // 享元池
    private final Map<String, Connection> pool = new ConcurrentHashMap<>();
    private final int maxSize;
    
    public ConnectionPool(int maxSize) {
        this.maxSize = maxSize;
    }
    
    public Connection getConnection(String config) {
        // 如果连接已存在，直接返回
        if (pool.containsKey(config)) {
            System.out.println("复用连接: " + config);
            return pool.get(config);
        }
        
        // 创建新连接
        if (pool.size() < maxSize) {
            Connection conn = createConnection(config);
            pool.put(config, conn);
            System.out.println("创建新连接: " + config);
            return conn;
        }
        
        throw new RuntimeException("连接池已满");
    }
    
    private Connection createConnection(String config) {
        // 模拟创建连接
        return new Connection(config);
    }
    
    public static class Connection {
        private final String config;
        
        public Connection(String config) {
            this.config = config;
        }
    }
}
```

### 7.3 Java 中的享元模式

```java
// Integer 缓存
Integer a = 100;
Integer b = 100;
System.out.println(a == b); // true，-128 到 127 之间的整数会被缓存

// String 常量池
String s1 = "hello";
String s2 = "hello";
System.out.println(s1 == s2); // true
```

---

## 八、模式选择指南

| 需求 | 推荐模式 |
|------|----------|
| 控制对象访问，添加额外功能 | 代理模式 |
| 接口不兼容，需要转换 | 适配器模式 |
| 动态添加功能，避免类爆炸 | 装饰器模式 |
| 树形结构，统一处理 | 组合模式 |
| 简化复杂系统调用 | 外观模式 |
| 多维度独立变化 | 桥接模式 |
| 大量相似对象，节省内存 | 享元模式 |

---

## 九、实战建议

1. **代理模式**：Spring AOP 已内置，优先使用注解方式
2. **适配器模式**：新旧系统迁移时非常有用
3. **装饰器模式**：Java IO 和 Servlet Filter 都是经典应用
4. **组合模式**：菜单、文件系统、组织架构等树形结构
5. **外观模式**：为复杂子系统提供简洁接口
6. **桥接模式**：多维度变化时避免类爆炸
7. **享元模式**：连接池、线程池是典型应用

---

> 💡 **提示**：结构型模式关注的是类与对象的组合，理解组合关系比理解继承更重要。
