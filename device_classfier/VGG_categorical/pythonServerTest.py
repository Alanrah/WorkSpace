# -*- coding: cp936 -*-
'''''
����һ��python server������ָ���˿ڣ�
����ö˿ڱ�Զ�����ӷ��ʣ����ȡԶ�����ӣ�Ȼ��������ݣ�
����������Ӧ������
'''
if __name__ == "__main__":
    import socket
print("Server is starting")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('localhost', 8002))  # ����soket����IP��ַ�Ͷ˿ں�
sock.listen(5)  # ������������������������Ӻ�server��ͨ����ѭFIFOԭ��,ָ�����������ٸ��ͻ����ӵ�������
print("Server is listenting port 8002, with max connection 5")

while True:  # ѭ����ѯsocket״̬���ȴ�����
    '''�� ��accept����ʱ��socket��ʱ�롰waiting��״̬���ͻ���������ʱ�������������Ӳ����ط�������accept��������һ����������Ԫ�ص� Ԫ��(connection,address)����һ��Ԫ��connection���µ�socket���󣬷���������ͨ������ͻ�ͨ�ţ��ڶ���Ԫ�� address�ǿͻ���Internet��ַ��'''
    connection, address = sock.accept()
    print(address)


    try:
        connection.settimeout(50)
        # ���һ�����ӣ�Ȼ��ʼѭ������������ӷ��͵���Ϣ
        ''''' 
        ���serverҪͬʱ���������ӣ������������Ӧ���ö��߳������� 
        ����server��ʼ�����������while�����ﱻ��һ��������ռ�ã� 
        �޷�ȥɨ�������������ˣ������̻߳�Ӱ�����ṹ�����Լǵ�������������1ʱ 
        ��������Ҫ��Ϊ���̼߳��ɡ� 
        '''
        while True:
            buf = connection.recv(1024).decode()
            print("Get value " + buf)
            if buf == '1':
                print("send welcome")
                connection.send('welcome to server!'.encode())
            elif buf == '2':
                connection.send('it is 2!'.encode())
                print( "send 2 back")
            else:
                print("close")

            #break  # �˳����Ӽ���ѭ��
    except socket.timeout:  # ����������Ӻ󣬸��������趨��ʱ���������ݷ�������time out
        print( 'time out')


   #print("closing one connection")  # ��һ�����Ӽ���ѭ���˳������ӿ��Թص�
    #connection.close()
