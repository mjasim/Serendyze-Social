<?php
 $str_json = file_get_contents('php://input');
 $username=$_GET['q'];
 $fp = fopen($username.'log.csv', 'a');
 fwrite($fp,$str_json);
 fclose($fp);
?>