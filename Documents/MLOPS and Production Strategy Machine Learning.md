**DevOps software development (dev) and operations (ops)** is a set of practices that combines software development (Dev) and IT operations (Ops). It aims to shorten the systems development life cycle and provide continuous delivery with high software quality. DevOps is complementary to Agile software development.

**Fundamentals of DevOps**:


**Code**: The first step in the DevOps life cycle is coding, where developers build the code on any platform.
**Build**: Developers build the version of their program in any extension, depending on the language they are using.
**Test**: For DevOps to be successful, the testing process must be automated using any automation tool like Selenium.
**Release**: A process for managing, planning, scheduling, and controlling the build in different environments after testing and before deployment.
**Deploy**: This phase gets all artifacts/code files of the application ready and deploys/executes them on the server.
**Operate**: The application is run after its deployment, where clients use it in real-world scenarios.
**Monitor**: This phase helps in providing crucial information that helps ensure service uptime and optimal performance.
**Plan**: The planning stage gathers information from the monitoring stage and, as per feedback, implements the changes for better performance.


**DevOps Lifecycle**:

Continuous Development
Continuous Integration
Continuous Testing
Continuous Monitoring
Virtualization and Containerization


**Continuous Development**

In the Waterfall model, our software product gets broken into multiple pieces or sub-parts to make the development cycles shorter, but in this stage of DevOps data science, the software is getting developed continuously.

**Continuous Integration**

In this stage, if our code is supporting new functionality, it is integrated with the existing code continuously. As the continuous development keeps on, the existing code needs to be integrated with the latest one continuously, and the changed code should ensure that there are no errors in the current environment for it to work smoothly.

**Continuous Testing**

In the continuous testing stage, our developed software is getting tested continuously to detect bugs using several automation tools.

Tools used: For QA/Testing purposes, we can use many automated tools, and the tool widely used for automation testing is Selenium, as it lets QAs test the code in parallel to ensure that there are no errors, inconsistencies, or flaws in the software.


**Continuous Monitoring**

It is a very crucial part of the DevOps life cycle, where it provides important information that helps us ensure service uptime and optimal performance. The operations team gets results from reliable monitoring tools to detect and fix the bugs/flaws in the application.


**Version Control System (VCS)** is a tool that helps track and manage changes to a project’s codebase over time.

It allows multiple developers to work on the same project simultaneously without conflicts, maintains a history of all changes, and enables easy rollback to previous versions if needed. VCS ensures collaboration, code integrity, and efficient management of software development.

1. **Track Changes Over Time**: VCS allows developers to track every modification made to the codebase. This means you can always go back to previous versions, ensuring no changes are lost.
2. **Collaboration**: VCS simplifies collaboration between team members. Everyone can work on different parts of the project without worrying about overwriting each other’s work.
3. **Code History and Audit Trails**: With version control, you can see who made specific changes, when they were made, and why. This audit trail is invaluable for debugging, reviewing, or maintaining code.
4. **Backup and Recovery**: Version control systems offer a way to back up your project. If something goes wrong, you can always recover previous versions.
5. **Branching and Merging**: You can create branches for different features or bug fixes, allowing multiple developers to work simultaneously without interfering with each other’s code. Once the work is done, branches can be merged back into the main project seamlessly.


Types of Version Control Systems

There are two main types of Version Control Systems: Centralized Version Control Systems (CVCS) and Distributed Version Control Systems (DVCS).

1. **Centralized Version Control Systems (CVCS)** operate with a single global repository, requiring users to commit their changes to make them part of the system. For others to see these changes, they must update their local copy. This model facilitates collaboration among developers by providing a shared view of project activity and offering administrators fine-grained control over permissions and actions. However, CVCS comes with drawbacks, most notably, the centralized repository creates a single point of failure. If the repository becomes unavailable, collaboration halts, and no versioned changes can be saved during the downtime. These limitations eventually led to the development of Distributed Version Control Systems (DVCS).

    Centralized Version Control Systems (CVCS) offer several advantages, including simplicity in setup and management, ease of maintaining a single central repository, and suitability for small teams or projects with limited collaboration needs. However, they also have notable disadvantages. If the central server goes down, no one can commit changes or retrieve updates, which can disrupt workflow. CVCS also provides limited support for branching and merging compared to Distributed Version Control Systems (DVCS), and the central server can become a bottleneck when many developers are committing changes simultaneously.





2. **Distributed Version Control Systems (DVCS)** operate with multiple repositories, giving each user their own repository and working copy. Unlike centralized systems, committing changes only updates the local repository; to share these changes with others, you must push them to a central or shared repository. Likewise, updating your working copy does not reflect others’ changes unless you first pull them into your local repository. For changes to be visible to others, four steps are necessary: you commit, you push, they pull, and they update. Popular DVCS tools like Git and Mercurial address the limitations of centralized systems, particularly by eliminating the single point of failure.

    Distributed Version Control Systems (DVCS) offer several advantages, such as enabling developers to work offline and commit changes locally before syncing with others. They handle branching and merging more effectively and provide faster access to version history, since developers do not rely on a central server. DVCS is also more resilient, as the repository can be recovered from any copy if one is lost. However, DVCS comes with some disadvantages, including a more complex setup and configuration, increased storage requirements due to each developer maintaining a full repository, and potentially higher bandwidth usage when pushing to or pulling from central servers.

Popular Version Control Systems:

1.**Git**

Git is the most widely used Distributed Version Control System, developed by Linus Torvalds in 2005 for managing the Linux kernel. It is highly efficient, supports branching and merging, and has a fast, decentralized workflow. Git is the backbone of services like GitHub, GitLab, and Bitbucket, making it a popular choice for developers worldwide.

Key Features of Git

- Lightweight, fast, and efficient.
- Branching and merging are simple and non-destructive.
- Provides powerful commands like git clone, git pull, and git push.

2.**Subversion (SVN)**

Subversion is a popular centralized version control system. While not as commonly used in open-source projects today, SVN is still used by many organizations and enterprises for its simplicity and centralized nature.

Key Features of SVN

- Single central repository.
- Supports branching and tagging, but is less flexible than Git.
- Versioning of files and directories

3.**Mercurial**

Mercurial is another distributed version control system similar to Git but with a simpler interface. It is well-suited for both small and large projects and is used by companies like Facebook and Mozilla.

Key Features of Mercurial

- Simple, fast, and scalable.
- Supports branching and merging.
- Includes tools for managing project history and changes.


Benefits of the version control system

Version control systems (VCS) offer several benefits that are crucial for effective collaboration and managing the lifecycle of software development projects. 

Below are the key advantages of using a VCS:

- **Enhanced Collaboration**: Multiple developers can work on the same project without conflicts by using branches and merging changes seamlessly.
- **Track Changes**: Every change made to the code is recorded with a detailed history, allowing easy rollback and audit trails.
- **Branching & Merging**: Developers can work on new features or fixes in isolated branches, which can be merged back into the main project later.
- **Backup & Recovery**: VCS ensures automatic backups, allowing recovery of previous versions if something goes wrong.
- **Improved Code Quality**: Code reviews and tracked changes help maintain quality and consistency.
- **Efficient Remote Collaboration**: Teams can work offline and sync later, enabling smooth collaboration across geographies.
- **Continuous Integration**: Automates testing and deployment, reducing errors and speeding up delivery.
- **Better Project Management**: Keeps track of milestones, commits, and progress, improving project visibility.
- **Security**: Provides role-based access and clear logs for security audits.
	

Use of Version Control System
Version control systems (VCS) are essential tools for managing changes to a project’s codebase, especially in collaborative environments.

- **Tracking Changes**: VCS tracks all changes made to the project, including who made the change, what was changed, and when it was made. This allows developers to monitor the evolution of the code and revert to previous versions if needed.
- **Collaboration**: VCS enables multiple developers to work on the same project simultaneously without overwriting each other’s work. By using branches, developers can work on different tasks or features independently and later merge their changes into the main project.
- **Conflict Resolution**: When changes made by different developers conflict, VCS tools help identify and resolve the conflicts, ensuring the code stays consistent and functional.
- **Backup and Recovery**: VCS acts as a backup system. If an error occurs, previous versions of the code can be recovered, minimizing the risk of data loss.
- **Testing Without Risk**: Developers can create isolated branches to test new features or bug fixes without affecting the main codebase. Once the changes are verified, they can be merged back into the main project.

- **Continuous Integration**: VCS supports continuous integration, allowing developers to regularly integrate their code changes, ensuring that the software remains in a deployable state at all times.


**Continuous integration (CI)** refers to the practice of automatically and frequently integrating code changes into a shared source code repository. Continuous delivery and/or deployment (CD) is a 2 part process that refers to the integration, testing, and delivery of code changes. Continuous delivery stops short of automatic production deployment, while continuous deployment automatically releases the updates into the production environment. Taken together, these connected practices are often referred to as a "CI/CD pipeline" and are supported by development and operations teams working together in an agile way with either a DevOps or site reliability engineering (SRE) approach.

	
CI/CD helps organizations avoid bugs and code failures while maintaining a continuous cycle of software development and updates. 

As apps grow larger, features of CI/CD can help decrease complexity, increase efficiency, and streamline workflows.

Because CI/CD automates the manual human intervention traditionally needed to get new code from a commit into production, downtime is minimized, and code releases happen faster. And with the ability to more quickly integrate updates and changes to code, user feedback can be incorporated more frequently and effectively, meaning positive outcomes for end users and more satisfied customers overall.



**Continuous delivery** automates the release of validated code to a repository following the automation of builds and unit and integration testing in CI. So, to have an effective continuous delivery process, CI must already be built into your development pipeline.

In continuous delivery, every stage—from the merger of code changes to the delivery of production-ready builds—involves test automation and code release automation. At the end of that process, the operations team can swiftly deploy an app to production.

**Continuous delivery** usually means a developer’s changes to an application are automatically bug-tested and uploaded to a repository (like GitHub or a container registry), where they can then be deployed to a live production environment by the operations team. It’s an answer to the problem of poor visibility and communication between dev and business teams. To that end, the purpose of continuous delivery is to have a codebase that is always ready for deployment to a production environment and ensure that it takes minimal effort to deploy new code.

**CI/CD** is an essential part of DevOps methodology, which aims to foster collaboration between development and operations teams. Both CI/CD and DevOps focus on automating processes of code integration, thereby speeding up the processes by which an idea (like a new feature, a request for enhancement, or a bug fix) goes from development to deployment in a production environment where it can provide value to the user.

In the collaborative framework of DevOps, security is a shared responsibility integrated from end to end. It’s a mindset that is so important, it led some to coin the term "DevSecOps" to emphasize the need to build a security foundation into DevOps initiatives. DevSecOps (development, security, and operations) is an approach to culture, automation, and platform design that integrates security as a shared responsibility throughout the entire IT lifecycle. A key component of DevSecOps is the introduction of a secure CI/CD pipeline.


**Containers** are a software package in a logical box with everything that the application needs to run. The software package includes an operating system, application code, runtime, system tools, system libraries, and binaries and etc.

Containers run directly within the Host machine kernels. They share the Host machine’s resources (like Memory, CPU, disks and etc.) and don’t need the extra load of a Hypervisor. This is the reason why Containers are “lightweight“. Containers are much smaller in size than a VM, and that is why they require less time to start, and we can run many containers on the same compute capacity as a single VM. This helps in high server efficiencies and therefore reduces server and licensing costs.

Container technology is almost as old as VMs, although the IT industry wasn’t employing containers until 2013-14 when Docker and Kubernetes, and other tech made waves that caused craziness in the industry. Containers have become a major trend in software development as an alternative or companion to Virtual machines. Containerization helps developers to create and deploy applications faster and more securely. Over the past few years, we’ve been teaching new technology- Docker & Kubernetes, and we cover all about Containers in detail because the rapid growth of Containers in the span of the last two decades has changed the dynamic of the modern IT industry. Containers are used for easily deploying Microservices applications. It’s difficult to talk about microservices without talking about containers.


1.**Lightweight**: Containers share the machine OS kernel and therefore don’t need a full OS instance per application. This makes the container files smaller, and this is the reason why Containers are smaller in size, especially compared to virtual machines. As they are lightweight, they can spin up quickly and can be easily scaled horizontally.
2.**Portable**: Containers are a package having all their dependencies with them, which means that we have to write the software once, and the same software can be run across different laptops, cloud, and on-premises computing environments without the need to configure the software again.
3.**Supports CI/CD**: Due to a combination of their deployment portability/consistency across platforms and their small size, containers are an ideal fit for modern development and application patterns, such as DevOps, serverless, and microservices.

4.**Improves utilization**: Containers enable developers and operators to improve CPU and memory utilization of physical machines.


Different Types of Containers:


1.**Docker** is one of the most popular and widely used container platforms. It enables the creation and use of Linux containers. Docker is a tool that makes the creation, deployment, and running of applications easier by using containers. Not only the Linux powers like Red Hat and Canonical embraced Docker, but companies like Microsoft, Amazon, and Oracle have also done it. Today, almost all IT and cloud companies have adopted Docker.

2.**LXC** is an open-source project of LinuxContainers.org. The aim of LXC is to provide isolated application environments that closely resemble virtual machines (VMs) but without the overhead of running their own kernel. LXC follows the Unix process model, in which there is no central daemon. So, instead of being managed by one central program, each container behaves as if it’s managed by a separate program. LXC works in a number of different ways from Docker. For example, we can run more than one process in an LXC container, whereas Docker is designed in such a way that running a single process in each container is better.

3.**CRI-O** is an open-source tool that is an implementation of the Kubernetes CRI (Container Runtime Interface) to enable using OCI (Open Container Initiative) compatible runtimes. Its goal is to replace Docker as the Container engine for Kubernetes. It allows Kubernetes to use any OCI-compliant runtime as the container runtime for running pods. Today, it supports runc and Kata Containers as the container runtimes, but any OCI-conformant runtime can be used.

4.**rkt** has a set of supported tools and a community to rival Docker. rkt containers, also known as Rocket, turn up from CoreOS to address security vulnerabilities in early versions of Docker. In 2014, CoreOS published the App Container Specification in an effort to drive innovation in the container space, which produced a number of open-source projects. Like LXC, rkt doesn’t use a central daemon and thereby provides more fine-grained control over your containers at the individual container level. However, unlike Docker, they’re not complete end-to-end solutions. But they are used with other technologies or in place of specific components of the Docker system.

5.**runC** is a lightweight universal OS container runtime. It was originally a low-level Docker component that worked under the hood, embedded within the platform architecture. However, it has since been rolled out as a standalone modular tool. The idea behind the release was to improve the portability of containers by providing a standardized, interoperable container runtime that can work both as part of Docker and independently of Docker in alternative container systems. As a result, runC can help you avoid being strongly tied to specific technologies, hardware, or cloud service providers.

6.**containerd** is basically a daemon, supported by both Linux and Windows, that acts as an interface between your container engine and container runtimes. It provides an abstracted layer that makes it easier to manage container lifecycles, such as image transfer, container execution, snapshot functionality, and certain storage operations, using simple API requests. Similar to runC, containerd is another core building block of the Docker system that has been separated off as an independent open-source project.



