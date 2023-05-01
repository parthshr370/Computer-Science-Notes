# Computer Networking Notes





_**TO LEARN HOW ALL COMPUTERS AND NETWORKS CONNECT AND WORK TOGETHER**_

1.  Based on client server architecture
2.  A system can act as a sever and a client
3.  `TCP` transmission control protocol means that the data will reach the correct person and fill data is sent
4.  `HTTP` is hyper text transfer protocol for web pages
5.  Data is sent in packets
6.  `UDP`, or User Datagram Protocol connection orientation less data without feedback from the ends
7.  `IP address` a unique address that identifies a device on the internet or a local network . 0-255 is the range
8.  `DNS` translates domain names into IP address
9.  `IP address` decides which device to send the data.`Port number` will denote the app requiring the data
10.  `Unguided` is like Bluetooth and Wi-Fi and `guided` is internet wire lines
11.  `SONET` synchronous optical cable
12.  `Frame reeling` connects local area network to the wider network
13.  `Modem` connects digital signal to electrical signal to transfer data
14.  `Topology` is the way structures communicates with each other
15.  `Open Systems Interconnection OSI` model was developed to create a standard for servers to communicate with each other
16.  `OSI` model has 7 layers `Application — presentation — session — transport — network — data link — physical`
17.  `Application layer` is implemented in software

-   `Presentation layer` - it takes the raw data to re presentable form - this includes encoding encryption extraction of data
-   `Session layer` - helps in setting up and managing the connection following the termination of the system . this just establishes the session and leaving the rest of the task on the next layer
-   `Transport layer` - it takes the data and does segmentation of the system and then transfers the data in the correct order
-   `Network layer` - the transmission of data from one computer to another that is in diff network . `Routers` live in this layer . Logical addressing is the ip addressing in networking . ip is given to each packet so right packet reaches right destination .
-   `Data link layer` - Allows you to directly communicate from the packet . The physical addressing (MAC address) is done in this layer .`MAC` addresses are added to the packets in a frame and then pushes the frame.
-   `Physical layer` - Physical routers are involved in this process

1.  `IP/TCP` model also called internet protocol suite developed by ARPA
2.  Layers are `Application — transport — network — data link — physical`
3.  `IP / TCP` models are more practical than the OSI model
4.  `Application` layer is the layer where users interact with the system . Browsers , WhatsApp etc are examples of application layer

-   `Client-Server` — Client sends a request to the server and server sends the response . Server controls the structure . Data centre is collection of huge number of servers
-   `Ping` measures the round trip time from host to the destination and then echoed back . We cannot reduce the the time of the ping since its the best possible speed
-   `P2P` is decentralised
-   `Repeater` is at the physical layer . its job is to regenerate the signal over the same network before the signal becomes too weak or corrupted . It extends the length to which the signal can be transmitted at the same network . Repeater does not amplify the signal it takes the weak signal and then copy the signal bit by bit and regenerate it at the original strength it at the original strength . Repeater is a 2 port device
-   `Hub` is a multi port repeater . It connects the wire in star structure at one point to connect diff stations . hub is a non intelligent device . They do not have intelligence to find the best data paths for the data packets.
-   `Active hub` have their own power supply and can clean boost and relay the signal along the network . It serves both as a repeater and writing centre
-   `Passive hub` are the hubs which collect wiring from the nodes and power supply from active hub . Does this without cleaning and boosting them . Cant be used un a long distance between nodes.
-   `Bridge` connects two LANs to make an extended LANs . `Transparent bridge` is when system is unaware of the existence of it in the system . It has two processes bridge forwarding and bridge learning . `Source Routing Bridges` in these bridges routing operation is performed by source station .
-   `Switch` is multi port bridge .It can even check error before sending the data and filter the data with errors
-   `Gateway` is used to connect two networks together . They basically work as the messenger agents .
-   `Protocols for web` `HTTP`(hyper text transfer protocol) — `DCHP`(Dynamic Host Configuration Protocol) — `FTP`(file transfer protocol) — `SMTP`(simple mail transfer protocol).
-   `SSH` to login to a terminal of someone else computer
-   `Telnet` is a terminal emulation that enables to connect remote or host to connect to a client using a telnet . Usually `port 23` is used . Typing `telnet hostname` will connect you to telnet client . Telnet is not encoded
-   `Thread` is a lighter version of a process . thread does one single job while process is whole
-   `Sockets` is used to send message from one sender to another
-   `Ports` tell which application are we using . `IP` tells us the device
-   `Ephemeral` ports are randomly generate temporary ports . They can exist on client side not on the server side
-   `HTTPs` is a client server protocol which tells us how you report the data from the server and also tells us how the server will send data to the client . HTTP uses TCP . HTTP is a state less protocol
-   `TCP` is connection oriented since we don’t have to lose any data we need to send .
-   `HTTP` methods `GET — POST — PUT — DELETE`
-   `Application layer protocols` require a transport layer protocol . `GET` to get something . `POST` to input a value
-   `Error Status report` `200` means successful `404` means couldn’t find and `500` means server error `300` means redirecting `400` means client error
-   `Cookies` is a unique string stores in the clients browser .Website sets up the cookie and when you log in next time then the server will send a request and the cookie stored in the browser . Server in will Know which data to send . 3rd party cookies are the ones which are stored from the websites which you do no visit
-   `Application protocol` for email is `SMTP (simple mail transfer protocol)` or `POP3`
-   `TCP` - transport layer protocol

1.  `How email work`

-   `Sender — senders SMTP server — receivers SMTP server — receiver`
-   POP - Port 110 `Client — connects and authorise — POP sever -- transact -- client`
-   `IMAP(internet message access protocol)` - allows us to view emails on multiple devices .Emails are stored in server while you delete them .Local copies of it are available on devices

1.  `Domain Name System (DNS)` - when you type the URL `www.google.com` ,HTTP protocol is going to take that domain name and its going to use DNS to convert that URL into the IP address and will connect to the server .DNS is a database service that is used to store IP address for all the domain names which are accessed by HTTP protocol
2.  `mail.google.com` mail(sub domain) . google(second level domain) . com(top level domain) . Instead of saving stuff in one database we store data in multiple databases.
3.  `Root DNS servers` are the first point of contact for query . they have top level domains (eg `io org com`)
4.  [rootservers.org](http://rootservers.org) for root DNS severs
5.  `.com` is for commercial `.edu` is for education
6.  `ICANN` maintains org and com
7.  `Local DNS` server is the first point of contact like ISP locally .If not found in the local DNS server then it looks in the root DNS server
8.  `Transport Layer` is inside the device .Its job is to take data from the network to the application an vice versa
9.  `Network Layer` lies outside the device . Transportation from network to network is done here . It takes care of delivering message from one computer to another but when the message is reached on the computer , the transport layer delivers it to the application .
10.  `Checksum` is a small-sized block of data derived from another block of digital data for the purpose of detecting errors that may have been introduced during its transmission or storage.
11.  Application layer `HTTP` — Transport layer `TCP/UDP` — Network layer `IP`
12.  `TCP` uses several `timers` to ensure that excessive delays are not encountered during communications.
13.  `Timer` terminates if there is no response from the user side and it once again starts the timer and send the packet to the receiver but it may cause duplicates to the receiver and he may get 2 packets we solve this problem using sequence number
14.  `Data` we send may not be in order , may change or may not be delivered .UDP uses checksums and will tell if the data has been corrupted or not
15.  `UDP` has source port number(2 bytes) — destination port number(2 bytes) — checksum(2 bytes) — length of datagram(2 bytes) all these terms are called headers and the total size is 8 bytes
16.  `DNS` uses `UDP` because its fast
17.  `Transport layer` checks if data doesn't arrive or maintains and arranges the data
18.  `65536` is the size of data you can send in one packet
19.  `Application layer` sends a lot of raw data and TCP breaks the data arranges it gives it headers and checksum
20.  `Congestion control` maintains the entry of data packets into the network
21.  `Full duplex` means data can be sent simultaneously in a single carries at the same time
22.  `Three segments` are exchanged between sender(client) and receiver(server) for a reliable TCP connection to get established , this is called 3 way handshake
23.  We generate random sequence numbers (SYN)
24.  `How 3 way handshake work`

-   Sequence number is generated from the senders side which tells that the client is gonna start a new communication request (SYN)
-   server responds to the request and produces SYN and ACK
-   In the final part client acknowledges the response of the server and they both establish a reliable connection with which they will start the actual data transfer

1.  We work with routers in `network layer`
2.  Every router has its own `network address`
3.  192.168.2.8 here 192.168.2 is the network address and .8 is the device address
4.  `Router` is a networking device that forwards data packets between computer network
5.  `Routers use *Routing Tables`* to determine out which interface the packet will be sent. A routing table lists all networks for which routes are known.
6.  Each router’s routing table is unique and stored in the `RAM` of the device.
7.  `Transport layer` is Data is in the form of segments
8.  `Network Data` is in the form of packets
9.  `Data link layer` Data is in the form of frames
10.  `Routers are nodes` links between the routers are the edges
11.  `Static routing` is done manually and is not adaptive
12.  `Routing` is the process of selecting a path for traffic in a network or between or across multiple networks.
13.  Network protocol is IP or internet protocol
14.  IP (defines a server a client ) v4 IS 32 bits and 4 words
15.  IPV6 is 128 bits
16.  Hopping happens in ISP
17.  A `subnet`, or `subnetwork`, is a [network](https://www.cloudflare.com/learning/network-layer/what-is-the-network-layer/) inside a network. `Subnets` make networks more efficient. Through `subnetting`, network traffic can travel a shorter distance without passing through unnecessary [routers](https://www.cloudflare.com/learning/network-layer/what-is-routing/) to reach its destination.
18.  `Subnet masking` makes the network part of the IP and will leave for us to use the host part
19.  12.0.0.0/31 means the first 31 bits are part of the `subnet` remains 1
20.  192.0.1.0/24 means 24 bits are occupied by the `subnet` remaining 8 bits
21.  Packets - header is of 20 bytes `IPv` , Length , Identification , Flag , protocols ,checksum , address , time to Live (TTL) etc
22.  `IPv4` has around 4.3 billion unique addresses
23.  `IPv6` is 4 times larger than IPv4
24.  Con of IPv6 is that its not backward compatible and would require a lot of effort to shift from IPv4 to IPv6 .A lot of devices would have to shift with a lot of hardwork
25.  A firewall establishes a barrier between secured internal networks and outside untrusted network, such as the Internet.
26.  `Firewall` is both both network layer and transport layer
27.  `Packets` recieved from network layer .Data link layer sends it over a physical link
28.  New device — DHCP server — Pool of IP address — Assigns IP to new device
29.  **`Network Address Translation (NAT)`** is a process in which one or more local IP address is translated into one or more Global IP address and vice versa in order to provide Internet access to the local hosts.
30.  `Onion routing` — in this case the message from the initial node travels through many nodes like an onion and travels till the very end when it reaches the end point it has traveled through many layers of encryption
31.  `Node 1` has address of `node 2` and key 1 so it encrypts the data with the key but realises that it has multiple layers of encryption ahead so it passes it to the next node
32.  When the data reaches the `final node (**exit node`)** peels off the last layer of encryption and finds a `**GET request**` for [youtube.com](http://youtube.com) and passes it onto the destination server.Server processes the request and shows you the result
33.  Each of these nodes has hundreds of concurrent connections going on, and to know which one leads to the right source and destination is not that easy.
34.  If someone knows your end point start point and the frequency of the request at the fraction of time then the person is exposed
35.  `MANET` is mobile Ad hoc network is where a device acts as a receiver and a sender . MANETs consist of a peer-to-peer, self-forming, self-healing network . This system is decentralized . Each node can play both the roles ie. of router and host showing autonomous nature.
36.  A switch forwards out the broadcast frame, generated by another device, to all its ports. If no loop avoidance schemes are applied then the switches will flood broadcasts endlessly throughout the network which consumes all the available bandwidth. This is called a broadcast storm. A broadcast storm is a serious network problem and can shut down the entire network in seconds.
37.  if host A sends a unicast frame for host B then switch A will receive the frame. Switch A will forward it out to both switch B and switch D which in turn both forward it out to switch C. Now, switch C will receive the frame on two different ports with the same source mac address therefore it will lead to instability in the MAC table in switch C. `Spanning Tree Protocol (STP)` is used to prevent these loops
38.  `SATA` (Serial AT Attachment) is a computer bus interface that connects host bus adapters to mass storage devices such as hard disk drives, optical drives
39. `DDoS` stands for `Distributed Denial of Service`, and it's a method where cybercriminals flood a network with so much malicious traffic that it cannot operate or communicate as it normally would. This causes the site's normal traffic, also known as legitimate packets, to come to a halt.

END OF THE DOCUMENT